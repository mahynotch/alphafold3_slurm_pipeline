# Examples

## af3pulldown Example

Predicting protein-protein interaction between two sequences:

1. Making features only:
```bash
af3pulldown --make_features \
  --bait_type protein --bait_input protein1.fasta \
  --prey_type protein --prey_input protein2.fasta \
  --destination ./features
```

2. Making complex predictions using existing features:
```bash
af3pulldown --make_complex \
  --bait_type protein --bait_input protein1.fasta \
  --prey_type protein --prey_input protein2.fasta \
  --destination ./output \
  --feature_path ./features
```

## af3oligomer Example

Predicting a complex of three different molecules:

1. Making features only:
```bash
af3oligomer --make_features \
  --input_type protein protein dna \
  --input protein1.fasta protein2.fasta dna1.fasta \
  --destination ./features
```

2. Making complex predictions using existing features:
```bash
af3oligomer --make_complex \
  --input_type protein protein dna \
  --input protein1.fasta protein2.fasta dna1.fasta \
  --destination ./output \
  --feature_path ./features
```

## af3monomer Example

Predicting structure for single protein sequences:

1. Predicting a single protein structure:
```bash
af3monomer \
  --job_name test_protein \
  --input protein.fasta \
  --destination ./output
```

2. Predicting multiple protein structures:
```bash
af3monomer \
  --job_name batch_prediction \
  --input protein1.fasta protein2.fasta protein3.fasta \
  --destination ./output 
```

## submit_input Example

Submit AlphaFold jobs directly using JSON configuration:

1. Submit a single prediction:
```bash
submit_input \
  --input input.json \
  --output ./output \
  --time 300 --mem 64
```

2. Submit multiple predictions from a directory:
```bash
submit_input \
  --input ./json_inputs \
  --output ./output \
  --time 600 --mem 128
```