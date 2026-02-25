import wfdb

record = wfdb.rdrecord("./test/japan/001")
fs = record.fs   
print(fs)
ann = wfdb.rdann("./test/japan/001", "atr")

print(ann.sample[:10])   # positions in samples
print(ann.symbol[:10])   # annotation codes
print(ann.aux_note[:10]) # rhythm labels (often AF info)