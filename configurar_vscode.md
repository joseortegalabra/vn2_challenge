## Readme de cómo configurar vscode Code para poder trabajar localmente en una mac utilizando este ambiente template


### 1. Instalar brew
Para poder instalar python en la mac se necesita instalar Brew (Brew, llamado oficialmente Homebrew, es un gestor de paquetes popular, principalmente para macOS. Permite instalar, actualizar, desinstalar y gestionar software desde la terminal de manera sencilla).

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```


### 2. Instalar libomp
Instalar librería libomp.dylib, necesaria para la paralelización en Mac(especialmente en chips Apple Silicon/M1/M2).

```bash
brew install libomp
```


### 3. Definir variables de entorno luego de instalar brew
Después de instalar brew debería aparecer la consola la recomendación de correr los siguientes comandos

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

### 4. Instalar pyenv
Install pyenv to easily manage Python versions. Follow the instructions according to the OS version. 

https://github.com/pyenv/pyenv?tab=readme-ov-file#installation


### 5. Instalar python
En el caso de este repo se trabaja con python 3.11, así que éste es instalado

```bash
pyenv install 3.11
pyenv local 3.11
```

Luego, asegurar que se tiene instalada python

```bash
python --version
python3 --version
```

### 6. Instalar poetry
Install Poetry to manage dependencies and virtual environments.
```bash
export POETRY_VERSION=1.8.3
curl -sSL https://install.python-poetry.org | python3 -
```

Then, Specify to poetry which python version to use

```bash
poetry env use 3.11
```

Finally, Install dependencies and project requirements
```bash
poetry install
```

Extra: Luego de tener instalado poetry se puede validar
```bash
poetry --version
```


### 7. Extra. Instalar extensión: jupyter
Permite visualizar y escribir notebooks como si se estuviera trabajar en jupyter notebooks/jupyter lab.

```python
# %%
var_a = 1
print(var_a)
```

### 8. Extra. activar co-pilot. Desactivar autocompletar de copilot
Cuando se escribe código, constantemente vscode (o copilot u otra extensión) recomienda código lo que puede ser molesto y desconcentrar más que ayudar.
Crear carpeta .vscode y al interior archivo settings.json (debe quedar de esta forma **.vscode/settings.json**)

```json
// settings.json
// ---- No mostrar sugerencias de autocompletado en línea
// idealmente buscar una forma que sugiera pero con delay alto
"editor.inlineSuggest.enabled": false
```


### 9. Extra. Instalar RUFF
Utilizar ruff como linter y formarter para validar código al momento de hacer el commit.
- Asegurarse que se tiene instalado **ruff (poetry)**
- Asegurarse que se tiene instalado **pre-commit (poetry)**
- Instalar **extensión ruff en vscode**
- Crear carpeta .vscode y al interior archivo settings.json (debe quedar de esta forma **.vscode/settings.json**)

```json
// settings.json
{
"python.formatter.provider": "ruff",
"editor.formatOnSave": true,
// Para que ruff también haga autofix de errores/lint
"ruff.lint.enable": true,
"ruff.lint.fixOnSave": true, // autofix errores al guardar
// Mostrar línea vertical - límite de caracteres permitidos
"editor.rulers": [
79
]
}
```
- Crear archivo **.pre-commit-config.yml** (ya creado en template repo con su contenido)
- correr el comando
```bash
poetry run pre-commit install
```