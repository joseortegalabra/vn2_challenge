"""
Vamos a hacerlo paso a paso: primero te muestro cómo entrenar un modelo por SKU usando reinforcement learning (usando Stable-Baselines3, por ejemplo con PPO o DDPG), y después cómo usar ese modelo ya entrenado para hacer inferencia (decidir cuánto pedir) sin conocer la demanda real.

🧩 Contexto del problema
Cada SKU tiene su propio entorno (InventoryEnv) con:
Inventario actual (inv_current)
Inventario en tránsito (in_transit_w1, in_transit_w2)
Forecast para semanas 1, 2, 3 (pueden ser 3 pronósticos distintos o con intervalos de confianza)
Acción: cantidad a pedir esta semana
Recompensa: penaliza exceso o falta de inventario según el costo.
"""

'''


"""
🧠 1. Definimos el entorno para un SKU
"""
import gym
import numpy as np
from gym import spaces


class InventoryEnv(gym.Env):
    def __init__(self, forecast, demand_real, costs):
        super(InventoryEnv, self).__init__()
        self.forecast = (
            forecast  # dataframe o array con forecast t+1, t+2, t+3
        )
        self.demand_real = demand_real  # real de test
        self.costs = costs  # dict con {'overstock': c1, 'understock': c2}
        self.t = 0

        # Estado: [inv_actual, in_transit_1, in_transit_2, fcast_mean, fcast_p90, fcast_p10]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Acción: cantidad a pedir
        self.action_space = spaces.Box(
            low=0, high=500, shape=(1,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.inv = np.random.uniform(50, 100)
        self.in_transit = [np.random.uniform(0, 30), np.random.uniform(0, 30)]
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        fcast = self.forecast[self.t]
        return np.array(
            [
                self.inv,
                self.in_transit[0],
                self.in_transit[1],
                fcast["mean"],
                fcast["p90"],
                fcast["p10"],
            ],
            dtype=np.float32,
        )

    def step(self, action):
        order_qty = action[0]
        self.in_transit.append(order_qty)

        # Avanza lead time
        arrived = self.in_transit.pop(0)
        self.inv += arrived

        # Consumo según demanda real
        demand = self.demand_real[self.t]
        sold = min(self.inv, demand)
        self.inv -= sold

        # Costos
        understock = max(0, demand - sold)
        overstock = max(0, self.inv)

        cost = -(
            self.costs["overstock"] * overstock
            + self.costs["understock"] * understock
        )

        self.t += 1
        done = self.t >= len(self.forecast)
        obs = self._get_obs() if not done else np.zeros(6)
        return obs, cost, done, {}


"""
🚀 2. Entrenamiento del modelo RL
Entrenamos un modelo PPO por SKU.
"""
from stable_baselines3 import PPO

# ejemplo de datos
forecast_data = [
    {"mean": 120, "p90": 150, "p10": 90},
    {"mean": 130, "p90": 160, "p10": 100},
    {"mean": 140, "p90": 180, "p10": 110},
]
demand_real = [125, 135, 145]
costs = {"overstock": 1.0, "understock": 3.0}

env = InventoryEnv(forecast_data, demand_real, costs)

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=50_000)
model.save("ppo_inventory_sku_001")


"""
🔍 3. Inferencia (sin evaluación)
Una vez entrenado, usás el modelo para decidir cuánto pedir, incluso sin saber la demanda real.
El modelo solo usa el estado observado (inventarios + forecasts).
"""
from stable_baselines3 import PPO

# cargar modelo entrenado
model = PPO.load("ppo_inventory_sku_001")

# inputs de inferencia (no se conoce demanda real)
obs = np.array(
    [
        80,  # inventario actual
        20,  # en tránsito semana 1
        15,  # en tránsito semana 2
        130,  # forecast medio
        160,  # forecast p90
        100,  # forecast p10
    ],
    dtype=np.float32,
)

# obtener acción óptima
action, _ = model.predict(obs, deterministic=True)
print(f"Cantidad a pedir esta semana: {action[0]:.2f}")


"""
⚙️ 4. Para múltiples SKUs
Puedes:
Entrenar un modelo por SKU (en paralelo o en loop).
Guardar cada modelo como ppo_inventory_sku_<id>.zip.
En inferencia, cargar el correspondiente según el SKU.
"""
for sku_id in skus:
    env = InventoryEnv(forecast_sku[sku_id], demand_real_sku[sku_id], costs)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=30_000)
    model.save(f"models/ppo_inventory_{sku_id}")


"""
🔮 Extensión: múltiples forecasts (manejo de incertidumbre)
Puedes añadir más pronósticos como features (forecast_lgbm, forecast_arima, forecast_nn, etc.).
También podrías incluir la varianza entre modelos como proxy de incertidumbre.
Ejemplo de observación extendida:
"""
self.observation_space = spaces.Box(
    low=0, high=np.inf, shape=(8,), dtype=np.float32
)
...
return np.array(
    [
        self.inv,
        self.in_transit[0],
        self.in_transit[1],
        fcast["mean"],
        fcast["p90"],
        fcast["p10"],
        fcast["arima"],
        fcast["lgbm"],
    ],
    dtype=np.float32,
)


'''
