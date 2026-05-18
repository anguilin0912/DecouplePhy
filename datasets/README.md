We can not provide the dataset files for legal reasons. Therefore, all files are empty in this directory and serve as templates. However, the datasets can be obtained by the individual publishers and transcribed into the IPAL format following the instructions here:

[https://github.com/fkie-cad/ipal_datasets](https://github.com/fkie-cad/ipal_datasets)

###### A short guide

1. Clone the [https://github.com/fkie-cad/ipal_datasets](https://github.com/fkie-cad/ipal_datasets) repository
2. Select a dataset, e.g., SWaT, for this example.
3. Obtain the raw dataset from the publisher. Links to the dataset are provided in the `ipal_datasets/README.md` table in the column `Link`.
4. Place the dataset files in the respective folder. For example, `ipal_datasets/SWaT/raw/[dataset files]`.
5. Execute the `transcribe.{py/sh}` script.
6. Copy the `attack.json` and `ipal/*` files to this directory.