@echo off
setlocal
cd /d "%~dp0"
title Grupo 10 - Analisis de sentimiento y toxicidad

echo ==========================================
echo GRUPO 10 - INICIO CON PYTHON 3.13
 echo ==========================================
echo.
python --version
if errorlevel 1 goto sin_python
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 13) else 1)"
if errorlevel 1 goto version_incorrecta

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 13) else 1)"
    if errorlevel 1 goto entorno_anterior
) else (
    echo.
    echo [1/3] Creando entorno virtual con Python 3.13...
    python -m venv ".venv"
    if errorlevel 1 goto error_instalacion
)

echo.
echo [2/3] Instalando o verificando las bibliotecas...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error_instalacion
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error_instalacion

echo.
echo [3/3] Abriendo la aplicacion...
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto error_ejecucion
goto fin

:sin_python
echo ERROR: No se encuentra el comando python.
echo Abre CMD y verifica: python --version
goto fin

:version_incorrecta
echo ERROR: Este iniciador esta preparado para Python 3.13.
echo Se detecto otra version. Abre CMD y comprueba: python --version
goto fin

:entorno_anterior
echo ERROR: Ya existe una carpeta .venv creada con otra version de Python.
echo Renombra .venv a .venv_anterior y vuelve a ejecutar este archivo.
goto fin

:error_instalacion
echo ERROR: Fallo la instalacion. Verifica tu conexion a internet.
echo Copia o toma una captura del mensaje de error mostrado mas arriba.
goto fin

:error_ejecucion
echo ERROR: La aplicacion se ha cerrado por un problema.
echo Copia o toma una captura del mensaje de error mostrado mas arriba.
goto fin

:fin
echo.
pause
endlocal
