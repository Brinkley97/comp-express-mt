from comet import download_model, load_from_checkpoint

model_path = download_model("masakhane/africomet-stl-1.1")
model = load_from_checkpoint(model_path)

# To DO: load all triplets in a folder, for each triplet, file run the comet evaluation
#  and log the results in a comet_metrics.json file
#
# adapt script to accept model from command line

model_output = model.predict(data, batch_size=8, gpus=1)
print(model_output)
