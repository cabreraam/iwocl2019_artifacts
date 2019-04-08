

version=$1

for i in {1..10}
do 
#./nw_harp_test 23040 10 $version
	./nw_svm 23040 10 1 ${version} 
#./nw_harp_test 21424 10 1 ${version} 
#./nw_harp_test 16 10 1 ${version} 
done
