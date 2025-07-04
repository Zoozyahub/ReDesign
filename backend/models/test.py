from generation import InteriorGenerator
from segmentation import FurnitureSegmenter
from search import FurnitureVectorSearch


if __name__ == '__main__':
    # generator = InteriorGenerator()
    segmenter = FurnitureSegmenter(model_type="coco")
    
    # results = generator.generate(
    #         input_image_path="./test3.jpg",
    #         user_prompt="loft style, green sofa, green armchari, wood minimalizm table in the center of picture",
    #         output_dir="./generated_interiors",
    #         seed=42,
    #         return_image=True,
    #     )
       
#     results_seg = segmenter.segment_image(results['image'])
    results_seg = segmenter.segment_image('C:/Interior/dataset/images/hoff_4.png')   
    segmenter.visualize_results(results_seg)
#     segmenter.save_segmented_objects(results_seg, output_dir="./segmented_furniture")


    searcher = FurnitureVectorSearch(db_path="./chroma_db", collection_name="furniture_clip")
    similar_items = searcher.find_similar_furniture(results_seg)
    print(similar_items)
 
    # Вывод результатов
    searcher.display_results(similar_items)
    