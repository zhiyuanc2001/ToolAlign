from datasets import load_dataset

dataset = load_dataset('json', data_files="your preference data dir/G123_train_data.json", cache_dir="hfdata_cache")
print(dataset)

dataset.save_to_disk('your save dir/G123_train_data_v3_hf')