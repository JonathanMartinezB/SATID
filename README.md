# SATID — Sistema de Alerta Temprana de Inundaciones y Desastres

**Prototipo académico funcional · versión 0.1.0 · Aquitania, Boyacá (Colombia)**

SATID transforma la formulación del proyecto de grado en una aplicación web que consulta variables meteorológicas por coordenadas, permite gestionar varias zonas, compila series históricas, registra evidencias de campo y muestra un nivel experimental de alerta en un mapa.

> **Importante:** este repositorio es un prototipo de investigación. No es un sistema de alerta oficial, no garantiza anticipación de inundaciones y no sustituye al IDEAM, UNGRD, Alcaldía, Bomberos ni a los protocolos de gestión del riesgo. Los umbrales iniciales son configurables y no están calibrados para Aquitania. Antes de cualquier uso operativo, se requiere validación con registros observados y concertación con los organismos de socorro.

## 1. ¿Qué incluye?

- Panel por zona con precipitación acumulada de 24 y 72 horas, precipitación pronosticada para las siguientes 24 horas, humedad superficial del suelo y variables meteorológicas.
- Niveles **Verde, Amarillo y Rojo** calculados con reglas transparentes y umbrales editables.
- Mapa interactivo con OpenStreetMap y Carto; marcadores para todas las zonas guardadas; búsqueda por lugar; alta manual por coordenadas o clic en el mapa; importación y exportación GeoJSON/CSV.
- Consulta histórica horaria para 2016–2024 (o rango elegido) mediante la API de Open-Meteo con el modelo ERA5-Land, y descarga de archivos CSV.
- Consulta opcional de caudal modelado GloFAS para una celda cercana. Se presenta como contexto y no determina el nivel de alerta.
- Registro manual de eventos, observaciones y fuentes documentales, con URL, fecha y notas para conservar la trazabilidad.
- Laboratorio Random Forest: entrenamiento opcional a partir de un CSV etiquetado por el usuario, con partición de prueba, precisión, recall y balanced accuracy. La aplicación no inventa registros ni etiquetas para entrenar.
- Enlaces de búsqueda documental por zona para IDEAM, Corpoboyacá, UNGRD, Datos Abiertos Colombia y Copernicus Browser.

## 2. Adaptación de datos por ausencia de acceso directo a ERA5 / Earth Engine

El prototipo evita depender de una cuenta de Google Earth Engine. Para el historial meteorológico se conecta directamente con **Open-Meteo Historical Weather API** y solicita explícitamente `models=era5_land`. Por tanto, no se está sustituyendo ERA5-Land por una observación local: se está accediendo a un producto de reanálisis basado en ERA5-Land a través de otro servicio web.

- ERA5-Land tiene resolución espacial aproximada de 0,1° (del orden de 9–11 km), frecuencia horaria y cobertura histórica desde 1950, según la documentación del proveedor.
- El valor de una celda de reanálisis es una estimación espacial. No equivale a la lluvia medida por un pluviómetro en una finca, quebrada o barrio.
- El endpoint de pronóstico consulta los modelos meteorológicos disponibles en Open-Meteo; no debe interpretarse como una observación validada de una estación local.
- La humedad del suelo solicitada corresponde a la capa de 0–7 cm del producto, no a una medición directa de saturación profunda del suelo.

### Fuentes integradas

| Fuente | Uso | Requisito / límite |
|---|---|---|
| [Open-Meteo Forecast](https://open-meteo.com/en/docs) | Lluvia reciente y pronosticada, temperatura, humedad relativa, viento y humedad del suelo disponible | API sin clave; uso gratuito no comercial sujeto a límites y atribución CC BY 4.0 |
| [Open-Meteo Historical Weather](https://open-meteo.com/en/docs/historical-weather-api) | Serie horaria histórica mediante ERA5-Land / ERA5 | API sin clave; resolución de rejilla y naturaleza de reanálisis deben documentarse |
| [Open-Meteo Flood / GloFAS](https://open-meteo.com/en/docs/flood-api) | Caudal fluvial modelado complementario | Resolución aproximada de 5 km; la celda de río más cercana puede no representar el cauce de interés |
| [OpenStreetMap](https://www.openstreetmap.org/) | Mapa base y cartografía comunitaria | Datos ODbL; mostrar atribución y respetar políticas de teselas |
| [Nominatim](https://nominatim.openstreetmap.org/) | Geocodificar una búsqueda manual de lugar | El servicio público limita el uso a 1 petición por segundo; la app consulta solo por acción del usuario |
| [Overpass API](https://overpass-api.de/) | Consultar cauces y cuerpos de agua mapeados cerca de una zona | Servicio comunitario sin SLA; puede no tener todos los cauces rurales |
| [Copernicus Data Space Browser](https://dataspace.copernicus.eu/browser/) | Buscar, visualizar y descargar Sentinel-1 y Sentinel-2 | Datos Copernicus abiertos; el Browser requiere cuenta gratuita para determinadas funciones |
| [NASA GIBS](https://nasa-gibs.github.io/gibs-api-docs/) | Fuente adicional de visualizaciones satelitales globales | Enlace de consulta; no se procesa automáticamente en el prototipo |

La aplicación no realiza scraping indiscriminado de resultados de buscadores ni copia noticias automáticamente. En su lugar, genera búsquedas web específicas para la zona y permite guardar la fuente revisada con URL, fecha y resumen. Esto deja una cadena de evidencia auditable y evita tratar un titular sin verificar como dato meteorológico.

## 3. Ejecutar localmente

Requisitos: Python 3.10 o superior y conexión a internet para consultar los servicios externos.

### Windows (PowerShell)

```powershell
cd SATID
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Linux / macOS

```bash
cd SATID
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

La aplicación abrirá normalmente en `http://localhost:8501`.


## 4. Uso de la aplicación

1. En el panel izquierdo selecciona **Aquitania - centro urbano** o crea otra zona mediante búsqueda o coordenadas.
2. Abre **Panel de alerta** y revisa las variables consultadas. La primera carga requiere conexión a internet.
3. Abre **Mapa y zonas** para añadir puntos con clic, editar coordenadas, consultar hidrografía OSM o exportar/importar GeoJSON.
4. En **Historial y registros**, consulta una serie histórica y guarda eventos confirmados, por ejemplo, desde la memoria histórica de Bomberos.
5. En **Fuentes y configuración**, abre búsquedas documentales y guarda enlaces revisados.
6. En **Laboratorio IA**, carga un dataset real y etiquetado para entrenar Random Forest.

### Formato CSV de entrenamiento Random Forest

El CSV debe contener exactamente estas variables (se permiten columnas adicionales, que serán ignoradas):

```csv
precip_24h_mm,precip_72h_mm,forecast_24h_mm,soil_moisture,elevation_m,label
```

- `precip_24h_mm`: precipitación de las 24 horas anteriores, en mm.
- `precip_72h_mm`: precipitación acumulada de las 72 horas anteriores, en mm.
- `forecast_24h_mm`: precipitación pronosticada para 24 horas, en mm (o una variable histórica equivalente documentada para hindcasting).
- `soil_moisture`: humedad volumétrica del suelo 0–7 cm, si está disponible.
- `elevation_m`: altitud en metros.
- `label`: `Verde`, `Amarillo` o `Rojo`, según una clasificación validada por la fuente del registro.

Se exigen al menos 30 filas válidas, dos clases y cinco registros por clase. Este mínimo solo permite ejecutar el experimento de software; no garantiza que los datos sean suficientes desde el punto de vista hidrológico. El conjunto de prueba es aleatorio y por ello sus métricas no equivalen al hindcasting temporal 2016–2024 descrito en el documento del proyecto.


## 6. Correspondencia con el documento de grado

| Elemento formulado | Implementación del prototipo |
|---|---|
| Conocimiento del riesgo | Zonas configurables, historial, observaciones y fuentes documentales |
| Monitoreo y redes telemáticas | Consulta por API de precipitación, humedad del suelo, pronóstico y caudal modelado |
| Difusión y comunicación | Dashboard por zona con niveles Verde, Amarillo y Rojo y exposición de limitaciones |
| Capacidad de respuesta | Registro de evidencia y documentación de fuentes para discusión con organismos locales |
| RF-01 / RF-02 | Descarga de ERA5-Land y análisis básico de lluvia/humedad; no se calcula NDVI/NDFI dentro de esta versión |
| RF-03 / RF-04 | Random Forest opcional con CSV etiquetado; aún no realiza clasificación automática de coberturas Sentinel |
| RF-05 / RF-06 / RF-07 | Niveles, dashboard y transparencia sobre limitaciones de los datos |
| RF-08 | El historial se puede descargar; la validación hindcast formal contra la memoria de Bomberos queda pendiente de reunir y vincular eventos reales |
| RNF-03 | CSV y GeoJSON para interoperabilidad |

**Diferencias deliberadas:** al no contar con el flujo de procesamiento Sentinel de Google Earth Engine en este prototipo, no se calculan NDVI ni NDFI, no se clasifica cobertura del suelo y no se declara un modelo hidrológico calibrado. El acceso a Copernicus Browser se deja como flujo de consulta/descarga independiente. Estos componentes pueden añadirse en una fase posterior con datos, credenciales y validación específicos.

## 7. Ejecutar pruebas

```bash
pip install pytest
pytest -q
```

## 8. Licencias y atribución

El código de este prototipo se distribuye bajo MIT. Los datos y servicios externos conservan sus propias licencias. Open-Meteo solicita atribución (CC BY 4.0); OpenStreetMap requiere atribución y está sujeto a ODbL y a políticas específicas de sus servicios de teselas y Nominatim; Copernicus y NASA GIBS tienen sus propias condiciones de uso. Consulta siempre los términos vigentes antes de desplegar públicamente o usar los datos en un producto comercial.

En **Historial y registros**, los registros con nivel `Verde`, `Amarillo` o `Rojo` pueden exportarse directamente al formato de entrenamiento, con altitud de la zona incluida cuando esté configurada. Los registros `Sin evento` y `No determinado` se excluyen de ese archivo de entrenamiento.
