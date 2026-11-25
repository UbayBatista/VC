# Autores

# Trabajo realizado

## Trabajo individual de cada miembro

# Ejecución del cuaderno

## Comandos para crear el environment con lo necesario:

```
conda env create -f environment.yml
conda activate VC_P5
```

Para poder ejecutar el cuaderno, el directorio debe tener la siguiente estructura:

```
P5
 ├── best_face_embedding_resnet18.pth
 ├── content
 │   └── datos
 │       └── dataset
 │           ├── identity_CelebA.txt
 │           └── img_align_celeba
 ├── deploy.prototxt.txt
 ├── environments.yml
 ├── gallery_embeddings.pt
 ├── README.md
 ├── res10_300x300_ssd_iter_140000.caffemodel
 └── VC_P5.ipynb
```

# Fuentes

## Conversaciones con la IA

- Para sobreponer la imagen del famoso sobre el frame: https://chatgpt.com/share/6b05e067-ece0-83eb-8341-fe86febccd56

- Para extraer la información de los landmarks de la cara en el primer filtro: https://share.gemini.google/hreKMcZP6LcJ

- Para controlar el parpadeo: https://share.gemini.google/LUzMzPup7coK

- Para anclar los dibujos a las caras: https://claude.ai/share/4d7dd9ed-3fc0-4d70-8aa1-152e76c1f8d9