# for pgcc hypers experments
# 2022.3.7@ Miracleyin
# run in conda env

# base bpr
# kill all python
#sudo ps -ef | grep "python tune.py" | cut -c 9-16|sudo  xargs kill -s 9

# run experiments
# run a model all seed if you can
# if memory out, less some experiments
python tune.py --model WGCN --dataset ecommerce --params wgcn2019 &
python tune.py --model WGCN --dataset ecommerce --params wgcn2020 &
python tune.py --model WGCN --dataset ecommerce --params wgcn2021 &
python tune.py --model WGCN --dataset ecommerce --params wgcn2022 &
python tune.py --model WGCN --dataset ecommerce --params wgcn2023 &








