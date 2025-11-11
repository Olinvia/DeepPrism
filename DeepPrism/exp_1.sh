for i in 0.005 0.006 0.007 0.008 0.009 0.010 0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019 0.020
do
    python test_mnist.py --nframes=4 --nhidden=32 --nlayers=1 --eps=$i --bound_method="lp" --seed=1000 --model_dir="saved/mnist_04f_032h_1l.pt"
    python test_mnist.py --nframes=4 --nhidden=32 --nlayers=2 --eps=$i --bound_method="lp" --seed=1000 --model_dir="saved/mnist_04f_032h_2l.pt"
    python test_mnist.py --nframes=4 --nhidden=32 --nlayers=3 --eps=$i --bound_method="lp" --seed=1000 --model_dir="saved/mnist_04f_032h_3l.pt"
    python test_mnist.py --nframes=4 --nhidden=64 --nlayers=1 --eps=$i --bound_method="lp" --seed=1000 --model_dir="saved/mnist_04f_064h_1l.pt"
    python test_mnist.py --nframes=4 --nhidden=128 --nlayers=1 --eps=$i --bound_method="lp" --seed=1000 --model_dir="saved/mnist_04f_128h_1l.pt"
    python test_mnist.py --nframes=7 --nhidden=32 --nlayers=1 --eps=$i --bound_method="lp" --seed=1000 --model_dir="saved/mnist_07f_032h_1l.pt"
    python test_mnist.py --nframes=4 --nhidden=32 --nlayers=1 --eps=$i --bound_method="opt" --seed=1000 --model_dir="saved/mnist_04f_032h_1l.pt"
    python test_mnist.py --nframes=4 --nhidden=32 --nlayers=2 --eps=$i --bound_method="opt" --seed=1000 --model_dir="saved/mnist_04f_032h_2l.pt"
    python test_mnist.py --nframes=4 --nhidden=32 --nlayers=3 --eps=$i --bound_method="opt" --seed=1000 --model_dir="saved/mnist_04f_032h_3l.pt"
    python test_mnist.py --nframes=4 --nhidden=64 --nlayers=1 --eps=$i --bound_method="opt" --seed=1000 --model_dir="saved/mnist_04f_064h_1l.pt"
    python test_mnist.py --nframes=4 --nhidden=128 --nlayers=1 --eps=$i --bound_method="opt" --seed=1000 --model_dir="saved/mnist_04f_128h_1l.pt"
    python test_mnist.py --nframes=7 --nhidden=32 --nlayers=1 --eps=$i --bound_method="opt" --seed=1000 --model_dir="saved/mnist_07f_032h_1l.pt"
done
