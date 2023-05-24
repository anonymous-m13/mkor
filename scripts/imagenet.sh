cd ./data
mkdir imagenet 
cd imagenet

mkdir train
cd train
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

mkdir val
cd val
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xvf ILSVRC2012_img_val.tar
rm -f ILSVRC2012_img_val.tar
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
sh valprep.sh
rm -f valprep.sh
cd ../


cd ../