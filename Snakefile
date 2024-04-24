rule all:
    input:
        "data/model/all_features/log_reg",
        "data/model/all_features/random_forest",
        "data/model/numeric_only/log_reg",
        "data/model/numeric_only/random_forest"

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
    input: 
        "configs/data_info/{feature_set}.yaml",
        input_path = rules.convert_data.output.converted_data
    output:
        report(
           "data/processed/{feature_set}/target_distr.png"
        ),
        preprocessed_files = expand(["data/processed/{feature_set}/{name}_features.npy", "data/processed/{feature_set}/{name}_target.npy"], name=["train", "test"], feature_set="{feature_set}"),
        class_labels = "data/processed/{feature_set}/class_labels.json"
    shell:
        "python ./preprocess.py input_data={input.input_path} out_dir=data/processed/{wildcards.feature_set} data_info={wildcards.feature_set}"

rule train:
    threads: 8
    input: 
        "data/processed/{feature_set}/class_labels.json",
        "configs/model/{model_name}.yaml"
    output:
        report(
            directory("data/model/{feature_set}/{model_name}"),
            patterns=["{name}.csv"],
            category="Final model"),
        report("data/model/{feature_set}/{model_name}/conf_matrix.png"),
    shell:
        "python ./train_model.py input_dir=data/processed/{wildcards.feature_set} out_dir=data/model/{wildcards.feature_set}/{wildcards.model_name} model={wildcards.model_name}"

