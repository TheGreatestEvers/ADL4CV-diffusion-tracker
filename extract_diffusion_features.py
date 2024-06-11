from algorithms.feature_extraction_loading import extract_diffusion_features


if __name__ == '__main__':
    extract_diffusion_features(input_dataset_paths={'davis': 'tapvid_davis/'}, output_dataset_path='features/', diffusion_model_path='text-to-video-ms-1.7b/', restrict_ncomponents=True, n_components=10)