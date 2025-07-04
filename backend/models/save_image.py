paths = ['C:/Interior/dataset/images/hoff_4.png', 'C:/Interior/dataset/images/hoff_5.png', 
            'C:/Interior/dataset/images/hoff_6.png', 'C:/Interior/dataset/images/hoff_7.png']

classes = ['armchair', 'couch', 'couch', 'couch']

furnitures_ids = ['11', '12', '13', '14']

links = ['https://hoff.ru/catalog/gostinaya/kresla/kreslo_scandica_skott_id8851657/?articul=80538589',
            'https://hoff.ru/catalog/gostinaya/divany/pryamye/divan_khangel_id8951451/?articul=80543041',
            'https://hoff.ru/catalog/gostinaya/divany/pryamye/divan_aster_m_id7870882/?articul=80423900',
            'https://hoff.ru/catalog/gostinaya/divany/pryamye/divan_krovat_kheppi_id10109025/?articul=80622885'
            ]
partners = [2, 2, 2, 2]

prices = [15999, 39999, 29999, 59999]

names = ['Кресло SCANDICA Скотт', 'Диван Хангель', 'Диван Астер-М', 'Диван-кровать Хэппи']

metadatas = []
for path, class_name, furniture_id in zip(paths, classes, furnitures_ids):
    metadata = {
    'class': class_name,
    'model': 'COCO',  # Предполагаем, что модель COCO
    'image': path,
    'id_furniture': furniture_id
    }
    metadatas.append(metadata)

searcher.add_images_to_database(image_paths=paths, metadatas=metadatas,)