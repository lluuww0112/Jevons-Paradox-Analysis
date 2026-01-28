"""
Task Mapper: Maps specific tasks to coarse-grained categories for analysis.

This module provides a dictionary that maps specific task descriptions
to broader, more general categories to facilitate trend analysis.
"""


def map_task_to_coarse(task):
    """
    Maps a specific task description to a coarse-grained category.
    
    Args:
        task (str): Specific task description (case-insensitive)
        
    Returns:
        str: Coarse-grained category name
    """
    task_lower = task.lower()
    
    # ========== Classification Tasks ==========
    if any(x in task_lower for x in ['classification', 'classify', 'categorization']):
        if 'object detection' in task_lower:
            return 'Object Detection'
        elif 'face' in task_lower:
            return 'Face Recognition/Detection'
        elif 'fine-grained' in task_lower or 'fine grain' in task_lower:
            return 'Fine-grained Classification'
        elif 'medical' in task_lower or 'radiology' in task_lower or 'mammogram' in task_lower or 'biomedical' in task_lower:
            return 'Medical Image Analysis'
        elif 'multi-label' in task_lower:
            return 'Multi-label Classification'
        else:
            return 'Image Classification'
    
    # ========== Person Re-identification (check before detection) ==========
    if any(x in task_lower for x in ['person re-identification', 'person reid', 'person search', 're-identification']):
        return 'Person Re-identification'
    
    # ========== Detection Tasks ==========
    if 'detection' in task_lower:
        if 'object detection' in task_lower:
            return 'Object Detection'
        elif 'face' in task_lower:
            return 'Face Recognition/Detection'
        elif 'action' in task_lower:
            return 'Action Recognition'
        elif 'saliency' in task_lower:
            return 'Saliency Detection'
        elif 'anomaly' in task_lower:
            return 'Anomaly Detection'
        elif 'text' in task_lower or 'scene text' in task_lower:
            return 'Text Detection/Recognition'
        elif 'medical' in task_lower:
            return 'Medical Image Analysis'
        else:
            return 'Object Detection'
    
    # ========== Recognition Tasks ==========
    if 'recognition' in task_lower:
        if 'action' in task_lower:
            return 'Action Recognition'
        elif 'face' in task_lower:
            return 'Face Recognition/Detection'
        elif 'person re-identification' in task_lower:
            return 'Person Re-identification'
        elif 'scene' in task_lower:
            return 'Scene Recognition'
        elif 'emotion' in task_lower:
            return 'Emotion Recognition'
        elif 'medical' in task_lower:
            return 'Medical Image Analysis'
        else:
            return 'Image Classification'
    
    # ========== Segmentation Tasks ==========
    if any(x in task_lower for x in ['segmentation', 'parsing']):
        if 'instance' in task_lower:
            return 'Instance Segmentation'
        elif 'semantic' in task_lower:
            return 'Semantic Segmentation'
        elif 'panoptic' in task_lower:
            return 'Panoptic Segmentation'
        elif 'referring' in task_lower:
            return 'Referring Expression Segmentation'
        elif 'video object' in task_lower:
            return 'Video Object Segmentation'
        elif '3d' in task_lower or 'point cloud' in task_lower:
            return '3D Segmentation'
        else:
            return 'Segmentation'
    
    # ========== Vision-Language Tasks ==========
    if any(x in task_lower for x in ['visual question answering', 'vqa', 'visual qa']):
        return 'Visual Question Answering'
    
    if 'image captioning' in task_lower or 'caption generation' in task_lower or 'captioning' in task_lower:
        return 'Image Captioning'
    
    if 'video captioning' in task_lower or 'dense video captioning' in task_lower:
        return 'Video Captioning'
    
    if any(x in task_lower for x in ['visual grounding', 'phrase grounding', 'referring expression', 'grounding']):
        if 'video' in task_lower:
            return 'Video Grounding'
        else:
            return 'Visual Grounding'
    
    if any(x in task_lower for x in ['cross-modal', 'image-text', 'text-image', 'image and text', 'text and image']):
        if 'retrieval' in task_lower:
            return 'Cross-modal Retrieval'
        else:
            return 'Cross-modal Understanding'
    
    if 'vision-language' in task_lower or 'vision and language' in task_lower or 'vision-language' in task_lower:
        return 'Vision-Language Understanding'
    
    if 'visual dialog' in task_lower or 'visual dialogue' in task_lower:
        return 'Visual Dialog'
    
    # ========== Generation Tasks ==========
    if 'generation' in task_lower or 'synthesis' in task_lower:
        if 'text-to-image' in task_lower or 'text to image' in task_lower or 't2i' in task_lower:
            return 'Text-to-Image Generation'
        elif 'text-to-video' in task_lower or 'text to video' in task_lower:
            return 'Text-to-Video Generation'
        elif 'image generation' in task_lower or 'image synthesis' in task_lower:
            return 'Image Generation'
        elif 'video generation' in task_lower:
            return 'Video Generation'
        elif '3d' in task_lower or 'point cloud' in task_lower:
            return '3D Generation'
        elif 'motion' in task_lower:
            return 'Motion Generation'
        else:
            return 'Generation'
    
    # ========== 3D Vision Tasks ==========
    if '3d' in task_lower or 'point cloud' in task_lower or '4d' in task_lower:
        if 'detection' in task_lower:
            return '3D Object Detection'
        elif 'reconstruction' in task_lower:
            return '3D Reconstruction'
        elif 'pose' in task_lower:
            return '3D Pose Estimation'
        elif 'segmentation' in task_lower:
            return '3D Segmentation'
        elif 'visual grounding' in task_lower:
            return '3D Visual Grounding'
        else:
            return '3D Vision'
    
    # ========== Video Understanding Tasks ==========
    if 'video' in task_lower:
        if 'action' in task_lower:
            return 'Video Action Recognition'
        elif 'classification' in task_lower:
            return 'Video Classification'
        elif 'object segmentation' in task_lower:
            return 'Video Object Segmentation'
        elif 'grounding' in task_lower:
            return 'Video Grounding'
        elif 'question answering' in task_lower or 'vqa' in task_lower:
            return 'Video Question Answering'
        elif 'captioning' in task_lower:
            return 'Video Captioning'
        elif 'retrieval' in task_lower:
            return 'Video Retrieval'
        else:
            return 'Video Understanding'
    
    # ========== Restoration/Enhancement Tasks ==========
    if any(x in task_lower for x in ['super-resolution', 'super resolution', 'denoising', 'inpainting', 
                                      'dehazing', 'deblurring', 'restoration', 'enhancement', 'deraining']):
        return 'Image Restoration'
    
    # ========== Learning Paradigms ==========
    if 'zero-shot' in task_lower or 'zero shot' in task_lower:
        return 'Zero-shot Learning'
    
    if 'few-shot' in task_lower or 'few shot' in task_lower or 'low-shot' in task_lower or 'one-shot' in task_lower:
        return 'Few-shot Learning'
    
    # ========== Navigation Tasks ==========
    if 'navigation' in task_lower:
        return 'Navigation'
    
    # ========== Pose Estimation ==========
    if 'pose estimation' in task_lower or 'pose detection' in task_lower or 'pose prediction' in task_lower:
        if '3d' in task_lower:
            return '3D Pose Estimation'
        else:
            return 'Pose Estimation'
    
    # ========== Tracking Tasks ==========
    if 'tracking' in task_lower:
        return 'Object Tracking'
    
    # ========== Scene Understanding ==========
    if 'scene graph' in task_lower:
        return 'Scene Graph Generation'
    
    if 'scene understanding' in task_lower or 'scene recognition' in task_lower:
        return 'Scene Understanding'
    
    # ========== Depth Estimation ==========
    if 'depth estimation' in task_lower or 'depth prediction' in task_lower:
        return 'Depth Estimation'
    
    # ========== Optical Flow ==========
    if 'optical flow' in task_lower:
        return 'Optical Flow'
    
    # ========== Style Transfer ==========
    if 'style transfer' in task_lower:
        return 'Style Transfer'
    
    # ========== Action-related Tasks ==========
    if 'action' in task_lower:
        if 'localization' in task_lower or 'detection' in task_lower:
            return 'Action Recognition'
        elif 'forecasting' in task_lower or 'prediction' in task_lower:
            return 'Action Prediction'
        elif 'quality' in task_lower:
            return 'Action Quality Assessment'
        else:
            return 'Action Recognition'
    
    # ========== Medical Tasks ==========
    if any(x in task_lower for x in ['medical', 'radiology', 'mammogram', 'biomedical', 'histopathology', 
                                      'medical report', 'radiology report', 'ophthalmic']):
        return 'Medical Image Analysis'
    
    # ========== Audio-Visual Tasks ==========
    if 'audio-visual' in task_lower or 'audiovisual' in task_lower:
        return 'Audio-Visual Understanding'
    
    # ========== Autonomous Driving ==========
    if any(x in task_lower for x in ['autonomous driving', 'self-driving', 'driving', 'trajectory prediction']):
        return 'Autonomous Driving'
    
    # ========== Feature Learning ==========
    if any(x in task_lower for x in ['feature extraction', 'feature learning', 'representation learning', 
                                      'embedding', 'feature generation']):
        return 'Feature Learning'
    
    # ========== Retrieval Tasks ==========
    if 'retrieval' in task_lower:
        if 'video' in task_lower:
            return 'Video Retrieval'
        elif 'image' in task_lower or 'sketch' in task_lower:
            return 'Image Retrieval'
        else:
            return 'Retrieval'
    
    # ========== Estimation Tasks ==========
    if 'estimation' in task_lower:
        if 'age' in task_lower:
            return 'Age Estimation'
        elif 'pose' in task_lower:
            return 'Pose Estimation'
        elif 'depth' in task_lower:
            return 'Depth Estimation'
        else:
            return 'Estimation'
    
    # ========== Matching Tasks ==========
    if 'matching' in task_lower or 'correspondence' in task_lower:
        return 'Matching/Correspondence'
    
    # ========== Other Specific Tasks ==========
    if 'question answering' in task_lower and 'visual' not in task_lower:
        return 'Question Answering'
    
    if 'machine translation' in task_lower or 'translation' in task_lower:
        return 'Translation'
    
    if 'emotion' in task_lower:
        return 'Emotion Recognition'
    
    if 'aesthetics' in task_lower or 'quality assessment' in task_lower:
        return 'Image Quality Assessment'
    
    # Default category for unmatched tasks
    return 'Other'


def create_task_mapping_dict():
    """
    Creates a comprehensive task mapping dictionary by reading all task files.
    
    Returns:
        dict: Dictionary mapping specific tasks to coarse-grained categories
    """
    task_mapping = {}
    
    # Read all task files
    for year in range(2018, 2026):
        try:
            with open(f'{year}_tasks.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    task = line.strip()
                    if task:
                        # Use original case for key, but map using lowercase
                        task_mapping[task] = map_task_to_coarse(task)
        except FileNotFoundError:
            continue
    
    return task_mapping


# Main task mapping dictionary
TASK_MAPPER = create_task_mapping_dict()


def get_coarse_category(task):
    """
    Get the coarse-grained category for a given task.
    
    Args:
        task (str): Specific task description
        
    Returns:
        str: Coarse-grained category
    """
    return TASK_MAPPER.get(task, map_task_to_coarse(task))


if __name__ == '__main__':
    # Print statistics
    from collections import Counter
    
    categories = Counter(TASK_MAPPER.values())
    print(f"Total tasks mapped: {len(TASK_MAPPER)}")
    print(f"\nCoarse categories (sorted by frequency):")
    for category, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")
