##Main Paper

#python -c "import utils.plot as p; p.plot_loss_from_csv('results/bert_frequency.csv', 'BERT Large Uncased', 'Wikipedia')"
#cp /media/hdd/Code/quantization/figures/dense/BERT\ Large\ Uncased/Wikipedia/BERT\ Large\ Uncased_loss.pdf figures/paper/inversion_frequency_bert_test_loss.pdf

#python -c "import utils.plot as p; p.plot_scalability_from_csv('results/bert_scalability.csv', 'BERT Large Uncased')"
#cp /media/hdd/Code/quantization/figures/scalability/BERT\ Large\ Uncased_scalability.pdf figures/paper/bert_scalability.pdf
#
#python -c "import utils.plot as p; p.plot_loss_from_csv('results/bert_phase_2.csv', 'BERT Large Uncased', 'Wikipedia')"
#cp /media/hdd/Code/quantization/figures/dense/BERT\ Large\ Uncased/Wikipedia/BERT\ Large\ Uncased_loss.pdf figures/paper/bert_loss.pdf
#cp /media/hdd/Code/quantization/figures/dense/BERT\ Large\ Uncased/Wikipedia/BERT\ Large\ Uncased_loss_time.pdf figures/paper/bert_loss_time.pdf
#
#python -c "import utils.plot as p; p.plot_iteration_costs()"
#cp /media/hdd/Code/quantization/figures/iteration_costs/inversion_freq_sensitivity_autoencoder_cifar100.pdf figures/paper/inversion_freq_sensitivity_autoencoder_cifar100.pdf
#cp /media/hdd/Code/quantization/figures/iteration_costs/inversion_freq_sensitivity_bert_large_uncased_wikipedia.pdf figures/paper/inversion_freq_sensitivity_bert_large_uncased_wikipedia.pdf
##
#python pruning.py --method dense --model autoencoder --plot_all --dataset cifar100
#cp /media/hdd/Code/quantization/figures/dense/autoencoder/cifar100/test_loss.pdf figures/paper/inversion_frequency_autoencoder_test_loss.pdf
#
#python pruning.py --method dense --model bert_large_uncased_nvidia --compare_optimizers --dataset wikicorpus_phase2
#cp /media/hdd/Code/quantization/figures/dense/bert_large_uncased_nvidia/wikicorpus_phase2/optimizer_comparison.pdf figures/paper/time_breakdown_bert_large_uncased.pdf
##
#python pruning.py --method dense --model resnet50 --compare_optimizers --dataset imagenet
#cp /media/hdd/Code/quantization/figures/dense/resnet50/imagenet/optimizer_comparison.pdf figures/paper/time_breakdown_resnet50.pdf


#Appendix
python pruning.py --method dense --model bert_large_cased_classification --plot_all --dataset imdb --target_accuracy 99.4
cp /media/hdd/Code/quantization/figures/dense/bert_large_cased_classification/imdb/training_accuracy.pdf figures/paper/imdb_bert_large_epoch.pdf
cp /media/hdd/Code/quantization/figures/dense/bert_large_cased_classification/imdb/training_accuracy_time.pdf figures/paper/imdb_bert_large_time.pdf

python pruning.py --method dense --model bert_base_cased_question_answering --plot_all --dataset squad --target_accuracy 99.53
cp /media/hdd/Code/quantization/figures/dense/bert_base_cased_question_answering/squad/training_accuracy.pdf figures/paper/squad_bert_base_epoch.pdf
cp /media/hdd/Code/quantization/figures/dense/bert_base_cased_question_answering/squad/training_accuracy_time.pdf figures/paper/squad_bert_base_time.pdf

python pruning.py --method dense --model alexnet --plot_all --dataset cifar100
cp /media/hdd/Code/quantization/figures/dense/alexnet/cifar100/training_accuracy.pdf figures/paper/cifar100_alexnet_epoch.pdf
cp /media/hdd/Code/quantization/figures/dense/alexnet/cifar100/training_accuracy_time.pdf figures/paper/cifar100_alexnet_time.pdf

python pruning.py --plot_eigenvalues
cp /media/hdd/Code/quantization/figures/eigenvalues/right_eigenvalues.pdf figures/paper/right_eigenvalues.pdf
cp /media/hdd/Code/quantization/figures/eigenvalues/right_condition_number.pdf figures/paper/right_condition_number.pdf

python -c "import utils.plot as p; p.plot_accuracy_from_csv('results/resnet50.csv', 'ResNet-50', 'ImageNet')"
cp /media/hdd/Code/quantization/figures/dense/ResNet-50/ImageNet/ResNet-50_acc.pdf figures/paper/resnet50_accuracy.pdf
cp /media/hdd/Code/quantization/figures/dense/ResNet-50/ImageNet/ResNet-50_acc_time.pdf figures/paper/resnet50_accuracy_time.pdf