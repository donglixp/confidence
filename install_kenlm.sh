PWD_DIR=$(pwd)

mkdir -p toolkit
cd toolkit
rm -rf kenlm
wget http://kheafield.com/code/kenlm.tar.gz
tar -xvzf kenlm.tar.gz
rm -rf kenlm.tar.gz
cd kenlm
mkdir -p build
cd build
cmake ..
make -j 4

cd $PWD_DIR
