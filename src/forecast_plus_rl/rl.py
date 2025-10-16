"""
Ejemplo CHAT GPT

Entrenar un modelo PPO (policy gradient) de stable-baselines3
para un solo SKU, donde el agente recibe información rica:
Observaciones (state):
Inventario actual disponible (I_t)
Inventario en tránsito 1 (llega próxima semana)
Inventario en tránsito 2 (llega en 2 semanas)
Tres forecast scenarios:
forecast_low, forecast_med, forecast_high
Sus respectivos std o intervalos de confianza
Acción (a_t):
Cantidad a pedir esta semana (acción continua → pedido entre 0 y Qmax)
Reward (r_t):
r_t = -(\text{holding_cost} * I_{end} + \text{stockout_cost} * /max(0, D - ventas) + \text{order_cost} * Q_t)


libs:
pip install stable-baselines3 gym numpy matplotlib
"""

'''



import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt


class InventoryEnvMultiForecast(gym.Env):
    """
    Entorno de inventario con lead time=2 semanas y 3 forecasts posibles (low, mid, high)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        holding_cost=0.1,
        stockout_cost=5.0,
        order_cost=0.0,
        lead_time=2,
        max_inventory=500,
        Qmax=200,
        demand_noise=0.1,
        horizon=12,
        seed=None,
    ):
        super().__init__()

        self.h = holding_cost
        self.p = stockout_cost
        self.c = order_cost
        self.lead_time = lead_time
        self.max_inventory = max_inventory
        self.Qmax = Qmax
        self.demand_noise = demand_noise
        self.horizon = horizon
        self.rng = np.random.RandomState(seed)

        # Observaciones:
        # [I, in_transit_1, in_transit_2, f_low, f_med, f_high, f_std]
        n_obs = 7
        self.observation_space = spaces.Box(
            low=0, high=max_inventory, shape=(n_obs,), dtype=np.float32
        )
        # Acción: cantidad a pedir (0..Qmax)
        self.action_space = spaces.Box(
            low=0, high=Qmax, shape=(1,), dtype=np.float32
        )

    def reset(self):
        self.week = 0
        self.I = self.rng.uniform(10, 50)
        self.in_transit = [
            self.rng.uniform(0, 10) for _ in range(self.lead_time)
        ]
        # generamos forecasts
        self.forecasts = self._generate_forecasts()
        return self._get_obs()

    def _generate_forecasts(self):
        """Genera forecast low, mid, high y std para cada semana"""
        base = 50 + 10 * np.sin(np.linspace(0, np.pi, self.horizon))
        std = np.abs(self.rng.normal(5, 2, size=self.horizon))
        low = base - std
        high = base + std
        mid = base
        return {"low": low, "mid": mid, "high": high, "std": std}

    def _get_obs(self):
        idx = min(self.week, self.horizon - 1)
        obs = np.array(
            [
                self.I,
                self.in_transit[0],
                self.in_transit[1],
                self.forecasts["low"][idx],
                self.forecasts["mid"][idx],
                self.forecasts["high"][idx],
                self.forecasts["std"][idx],
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action):
        # llega orden del pipeline
        arrivals = self.in_transit[0]
        self.I += arrivals
        # mover pipeline
        self.in_transit[0] = self.in_transit[1]
        self.in_transit[1] = float(np.clip(action, 0, self.Qmax))

        # demanda real (muestreada de forecast medio ± ruido)
        mu = self.forecasts["mid"][self.week]
        sigma = self.forecasts["std"][self.week]
        demand = max(0, self.rng.normal(mu, self.demand_noise * sigma))
        sales = min(self.I, demand)
        unserved = max(0, demand - self.I)
        self.I -= sales

        # calcular costos
        cost = self.h * self.I + self.p * unserved + self.c * action
        reward = -float(cost)

        # avanzar semana
        self.week += 1
        done = self.week >= self.horizon
        info = {
            "demand": demand,
            "sales": sales,
            "unserved": unserved,
            "reward": reward,
        }

        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        print(
            f"Week={self.week}, Inventory={self.I:.1f}, Pipeline={self.in_transit}"
        )


# ---------- ENTRENAMIENTO POR SKU ----------
def train_sku(sku_id, episodes=200_000):
    env = InventoryEnvMultiForecast(seed=sku_id + 42)
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.001,
        n_epochs=10,
        tensorboard_log=f"./logs_sku_{sku_id}/",
    )

    model.learn(total_timesteps=episodes)
    model.save(f"ppo_inventory_sku_{sku_id}")
    return model, env


# ---------- EVALUACIÓN ----------
def evaluate_model(model, env, episodes=10):
    rewards, inventories, demands = [], [], []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        inv_track, dem_track = [], []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            inv_track.append(env.I)
            dem_track.append(info["demand"])
        rewards.append(total_reward)
        inventories.append(inv_track)
        demands.append(dem_track)
    return rewards, inventories, demands


if __name__ == "__main__":
    # Entrenamos un modelo por SKU
    models, envs = {}, {}
    sku_ids = [1, 2, 3]  # Ejemplo: 3 productos distintos
    for sku in sku_ids:
        print(f"\n🔹 Entrenando modelo para SKU {sku}")
        model, env = train_sku(sku, episodes=50_000)
        models[sku] = model
        envs[sku] = env

    # Evaluación
    for sku in sku_ids:
        model, env = models[sku], envs[sku]
        rewards, inventories, demands = evaluate_model(model, env, episodes=5)
        print(f"SKU {sku}: Reward promedio = {np.mean(rewards):.2f}")
        plt.figure(figsize=(8, 4))
        plt.plot(np.mean(inventories, axis=0), label="Inventario Promedio")
        plt.plot(np.mean(demands, axis=0), label="Demanda Promedio")
        plt.title(f"SKU {sku}")
        plt.legend()
        plt.show()


"""

Cada SKU entrena su propio agente PPO independiente.
Cada agente observa:
[inventario, en_transito_1, en_transito_2,
 forecast_low, forecast_mid, forecast_high, forecast_std]
El agente decide una acción continua = cuántas unidades pedir.
El entorno simula llegada, consumo y costos.
"""



'''
