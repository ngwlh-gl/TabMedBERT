task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    'cdr': ['Chemical','Disease'],
    'ade': ['Adverse-Effect','Drug'],
    'chr': ['ChemMet'],
    'biored':['GeneOrGeneProduct','DiseaseOrPhenotypicFeature','ChemicalEntity','OrganismTaxon','SequenceVariant','CellLine'],
    'ddi':['DRUG'],
    'gad':['GENE','DISEASE'],
    'chemprot':['GENE','CHEMICAL'],
    'ppi':['PROTEIN'],
    'drugprot':['CHEMICAL','GENE']
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
    'cdr':['CID'],
    'ade':['AE'],
    'chr': ['React'],
    'biored':['Association','Positive_Correlation','Bind','Negative_Correlation','Comparison','Drug_Interaction','Cotreatment','Conversion'],
    'ddi':['advise','mechanism','effect','int'],
    'gad':['relation'],
    'chemprot':['CPR:3','CPR:4','CPR:5','CPR:6','CPR:9'],
    'ppi':['relation'],
    'drugprot':["ACTIVATOR", "AGONIST", "AGONIST-ACTIVATOR", "AGONIST-INHIBITOR", "ANTAGONIST", "DIRECT-REGULATOR", "INDIRECT-DOWNREGULATOR", "INDIRECT-UPREGULATOR", "INHIBITOR", "PART-OF", "PRODUCT-OF", "SUBSTRATE", "SUBSTRATE_PRODUCT-OF"]
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
