rule all:
    input:
        "data/model"

rule convert_data:
    threads: 4
    input:
        raw_data = "data/raw/2015-street-tree-census-tree-data.csv"
    output:
        converted_data = "data/converted/2015-street-tree-census-tree-data.parquet"
    shell:
        "python ./read_data.py data_convert.input_path={input.raw_data} data_convert.out_path={output.converted_data}"


rule preprocess:
    threads: 4
    input: input_path = rules.convert_data.output.converted_data
    output:
        report(
           "data/processed/target_distr.png"
        ),
        preprocessed_files = expand(["data/processed/{name}_features.npy", "data/processed/{name}_target.npy"], name=["train", "test"]),
        class_labels = "data/processed/class_labels.json"
    shell:
        "python ./preprocess.py input_data={input.input_path} out_dir=data/processed"

rule train:
    threads: 8
    input: 
        rules.preprocess.output.preprocessed_files,
        rules.preprocess.output.class_labels
    output:
        report(
            directory("data/model"),
            patterns=["{name}.csv"],
            category="Final model"),
        report("data/model/conf_matrix.png")
    shell:
        "python ./train_model.py input_dir=data/processed out_dir=data/model"

