import pandas as pd 

clinical = pd.read_csv("/home/cai/naomi/immuneproject/table_s1clinical.csv")
genes = pd.read_csv("/home/cai/naomi/immuneproject/table_s13tpm.csv")

gene_patient_ids = genes.columns[2:]

#keeping only clinical patients that have gene expression data
clinical_rna = clinical[
    clinical["Harmonized_SU2C_RNA_Tumor_Sample_ID_v2"].isin(gene_patient_ids)
].copy()

#keeping only valid BOR responses
yravi = clinical_rna[
    clinical_rna["Harmonized_Confirmed_BOR"].isin(["PD", "SD", "PR", "CR"])
].copy()

yravi = yravi[
    [
        "Harmonized_SU2C_Participant_ID_v2",
        "Harmonized_SU2C_RNA_Tumor_Sample_ID_v2",
        "Harmonized_Confirmed_BOR"
    ]
]

#converting bor to labels
yravi["class_label"] = yravi["Harmonized_Confirmed_BOR"].map({
    "PD": 0,
    "SD": 0,
    "PR": 1,
    "CR": 1
})

print("Final patients:", yravi.shape)
print(yravi["Harmonized_Confirmed_BOR"].value_counts())

yravi.to_csv("Yravi.csv", index=False)

print("patients:", yravi.shape[0])
print(yravi["Harmonized_Confirmed_BOR"].value_counts())




#geetting RNA tumor sample IDs corresponding to the patients
rna_ids = yravi["Harmonized_SU2C_RNA_Tumor_Sample_ID_v2"].tolist()

print("RNA samples found:", len(rna_ids))

# gene expression matrix
Xravi = genes[["Name", "Description"] + rna_ids].copy()

#remove genes with mean expression < 1
Xravi_filtered = Xravi[
    Xravi[rna_ids].mean(axis=1) >= 1
].copy()

print("genes before filtering:", Xravi.shape[0])
print("genes after filtering:", Xravi_filtered.shape[0])

Xravi_filtered.to_csv("Xravi.csv", index=False)
