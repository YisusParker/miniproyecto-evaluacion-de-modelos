# Proyecto Integrador de Aprendizaje Automático (9.10)

Clasificación binaria (default en préstamos Lending Club), comparación scikit-learn vs PySpark e interpretabilidad con LIME. Entregable en formato Jupyter Book.

**Libro publicado:** [GitHub Pages](https://yisusparker.github.io/miniproyecto-evaluacion-de-modelos/) (tras activar Pages y hacer push).

## Cómo construir el libro en local

```bash
python -m venv .venv
source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter-book build .
```

Abre `_build/html/index.html` en el navegador.

## Despliegue en GitHub Pages

El workflow `.github/workflows/deploy-book.yml` construye el libro y lo publica en cada push a `main`. Solo hay que:

1. **Settings → Pages** del repositorio → **Build and deployment** → **Source**: **GitHub Actions**.
2. Hacer push a `main` (o ejecutar el workflow manualmente en la pestaña Actions).
