# split weight file
mkdir weight_split
split -b 90m ave_dense_544_final.weights weight_split/
# re-combine weight file
cat weight_split/* > ave_dense_544_final.weights