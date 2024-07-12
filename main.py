import shutil, csv, random
from torch.utils.data import DataLoader
from data.dataloader_paired import dataloader as DatasetLoader_SIDD
from data.dataloader_single import dataloader as DatasetLoader_DF2K
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from train import *
import utils as utils
import model as architecture
from option import args
from thop import profile_origin

import pynvml

def main():
    global opt
    opt = utils.print_args(args)

    if torch.cuda.is_available() and opt.cuda:
        opt.device = torch.device("cuda:0")
    else:
        opt.device = torch.device('cpu')
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    cudnn.benchmark = True

    print("===> Building model")
    model = {
        'Condformer': architecture.Condformer(inp_channels=3,
                                              out_channels=3,
                                              dim=48,
                                              num_blocks=[4, 6, 6, 8],
                                              num_refinement_blocks=4,
                                              heads=[1, 2, 4, 8],
                                              ffn_expansion_factor=2.66,
                                              bias=False,
                                              LayerNorm_type='BiasFree',
                                             ),
        'LoNPE': architecture.LoNPE(inp_channels=3, num_conditions=2, patch_size=32)
    }
    for item in model:
        if len(opt.GPUs) > 1:
            model[item] = nn.DataParallel(model[item], device_ids=opt.GPUs)
        model[item] = model[item].to(opt.device)

    for item in model:
        model[item] = utils.load_checkpoint(opt.resume[item], model[item], opt.GPUs, is_cuda=opt.cuda)

    if opt.train.lower() == 'train':

        optimizer = {'Condformer': None, 'LoNPE': None}
        scheduler = {'Condformer': None, 'LoNPE': None}

        print("===> Setting GPU")
        for item in model:
            para = filter(lambda x: x.requires_grad, model[item].parameters())
            optimizer[item] = opt.optimizer[item]([{'params': para, 'initial_lr': opt.lr[item]}],
                                                  lr=opt.lr[item], betas=(0.9, 0.999), weight_decay=1e-4)
            scheduler[item] = utils.learning_rate_scheduler(optimizer[item], opt.lr_type[item], opt.start_epoch,
                                                            opt.lr_gamma_1[item], opt.lr_gamma_2[item])
            model[item].train()
        start_epoch = opt.start_epoch if opt.start_epoch >= 0 else 0

        if os.path.exists(opt.model_path + '/' + 'runs'):
            shutil.rmtree(opt.model_path + '/' + 'runs')
        writer = SummaryWriter(opt.model_path + '/runs')

        print('===> Building Training dataloader')
        if 'SIDD_Medium_Srgb' in opt.data_train:
            trainset = DatasetLoader_SIDD(opt)

            print('===> Evaluation on SIDD validation dataset')
            result_path = opt.model_path + '/Results/SIDD Validation'
            os.makedirs(result_path)
            if opt.save_img and not os.path.exists(result_path + '/Images'):
                os.makedirs(result_path + '/Images')
                os.makedirs(result_path + '/Images_Norm')
            with open(result_path + '/Criteria.csv', 'w', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(['image_name', 'PSNR', 'Time', 'Sigma_s', 'Sigma_r'])
                PSNR, SSIM, NoiP, Time = valid_SIDD(opt.dir_data + 'Test/{}'.format(opt.data_test[0]),
                                                    result_path, model, opt, f_csv)
            writer.add_scalar('Validation_{}/PSNR'.format(opt.data_test[0]), PSNR, 0)
            writer.add_scalar('Validation_{}/DiffNoise'.format(opt.data_test[0]), NoiP, 0)

        elif 'DF2K' in opt.data_train:

            sigma_s = [np.array([0]), np.array([0.15]), np.array([0.3])]
            sigma_r = [np.array([15 / 255]), np.array([25 / 255]), np.array([50 / 255])]

            trainset = DatasetLoader_DF2K(opt)
            print('===> Evaluation on Poisson-Gaussian Synthetic Datasets')
            for data_test in opt.data_test[:1]:
                for sigma_s_ in sigma_s:
                    Avg_PSNR, Avg_SSIM = 0, 0
                    count = 0
                    for sigma_r_ in sigma_r:
                        print('===> Validation on {} with sigma_S={} and sigma_C={}'.format(data_test, sigma_s_,
                                                                                            sigma_r_))
                        result_path = opt.model_path + '/Results/{}/S{}C{}'.format(data_test, float(sigma_s_),
                                                                                   int(sigma_r_ * 255))
                        if not os.path.exists(result_path):
                            os.makedirs(result_path)
                        prior = np.append(sigma_s_, sigma_r_, axis=0)

                        with open(result_path + '/Criteria.csv', 'w', newline='') as f:
                            f_csv = csv.writer(f)
                            f_csv.writerow(['image_name', 'PSNR', 'Time', 'Sigma_s', 'Sigma_r'])
                            PSNR, SSIM, NoiP, Time = valid_PGNoise(opt.dir_data + 'Test/{}/HR'.format(data_test),
                                                                   result_path, model, prior, opt, f_csv)
                        Avg_PSNR += PSNR
                        count += 1
                        writer.add_scalar('Validation_{}/S{}C{}_PSNR'.
                                          format(data_test, float(sigma_s_), int(sigma_r_ * 255)),
                                          PSNR, 0)
                        writer.add_scalar('Validation_{}/S{}C{}_DiffNoise'.
                                          format(data_test, float(sigma_s_), int(sigma_r_ * 255)),
                                          NoiP, 0)
                    print('Avg PSNR on {} with sigma_S={} = {:.2f}'.format(data_test, sigma_s_, Avg_PSNR / count))
        else:
            raise InterruptedError


        for epoch in range(start_epoch + 1, opt.n_epochs + 1):
            writer.add_scalar('Learning Rate/Condformer', optimizer['Condformer'].param_groups[0]['lr'], epoch)
            writer.add_scalar('Learning Rate/LoNPE', optimizer['LoNPE'].param_groups[0]['lr'], epoch)

            if opt.shuffle:
                print('===> Building Training dataloader')
                if 'SIDD_Medium_Srgb' in opt.data_train:
                    trainset = DatasetLoader_SIDD(opt)
                elif 'DF2K' in opt.data_train:
                    trainset = DatasetLoader_DF2K(opt)
                else:
                    raise InterruptedError

            train_dataloader = torch.utils.data.DataLoader(trainset, num_workers=opt.threads, shuffle=True,
                                                           batch_size=opt.batch_size)
            print('===> Training Denoising Condformer')
            if 'SIDD_Medium_Srgb' in opt.data_train:
                train_SIDD(train_dataloader, optimizer, model, epoch, writer, opt)
                print('===> Evaluation on SIDD validation dataset')
                result_path = opt.model_path + '/Results/SIDD Validation'
                with open(result_path + '/Criteria.csv', 'w', newline='') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(['image_name', 'PSNR', 'Time', 'Sigma_s', 'Sigma_r'])
                    PSNR, SSIM, NoiP, Time = valid_SIDD(opt.dir_data + 'Test/{}'.format(opt.data_test[0]), result_path,
                                                  model, opt, f_csv)
                writer.add_scalar('Validation_{}/PSNR'.format(opt.data_test[0]), PSNR, epoch)
                writer.add_scalar('Validation_{}/DiffNoise'.format(opt.data_test[0]), NoiP, epoch)
            elif 'DF2K' in opt.data_train:
                train_DF2K(train_dataloader, optimizer, model, epoch, writer, opt)
                print('===> Evaluation on Poisson-Gaussian Synthetic Datasets')
                for data_test in opt.data_test[:1]:
                    for sigma_s_ in sigma_s:
                        Avg_PSNR, Avg_SSIM = 0, 0
                        count = 0
                        for sigma_r_ in sigma_r:
                            print('===> Validation on {} with sigma_S={} and sigma_C={}'.format(data_test, sigma_s_,
                                                                                                sigma_r_))
                            result_path = opt.model_path + '/Results/{}/S{}C{}'.format(data_test, float(sigma_s_),
                                                                                       int(sigma_r_ * 255))
                            if not os.path.exists(result_path):
                                os.makedirs(result_path)
                            prior = np.append(sigma_s_, sigma_r_, axis=0)

                            with open(result_path + '/Criteria.csv', 'w', newline='') as f:
                                f_csv = csv.writer(f)
                                f_csv.writerow(['image_name', 'PSNR', 'Time', 'Sigma_s', 'Sigma_r'])
                                PSNR, SSIM, NoiP, Time = valid_PGNoise(opt.dir_data + 'Test/{}/HR'.format(data_test),
                                                                       result_path, model, prior, opt, f_csv)
                            Avg_PSNR += PSNR
                            count += 1
                            writer.add_scalar('Validation_{}/S{}C{}_PSNR'.
                                              format(data_test, float(sigma_s_), int(sigma_r_ * 255)),
                                              PSNR, epoch)
                            writer.add_scalar('Validation_{}/S{}C{}_DiffNoise'.
                                              format(data_test, float(sigma_s_), int(sigma_r_ * 255)),
                                              NoiP, epoch)
                        print('Avg PSNR on {} with sigma_S={} = {:.2f}'.format(data_test, sigma_s_, Avg_PSNR / count))

            scheduler['Condformer'].step()
            scheduler['LoNPE'].step()

            model_path = opt.model_path + '/Checkpoints/Condformer_epoch_{}.pth'.format(epoch)
            torch.save(model['Condformer'].state_dict(), model_path)
            print('Checkpoint saved to {}'.format(model_path))

            model_path = opt.model_path + '/Checkpoints/LoNPE_epoch_{}.pth'.format(epoch)
            torch.save(model['LoNPE'].state_dict(), model_path)
            print('Checkpoint saved to {}'.format(model_path))

            torch.cuda.empty_cache()
        writer.close()
    elif opt.train.lower() == 'test':
        if 'SIDD' in opt.data_test:
            print('===> Evaluation on SIDD validation dataset')
            result_path = opt.model_path + '/Results/SIDD Validation'
            if not os.path.exists(result_path + '/Images'):
                os.makedirs(result_path + '/Images')
            if not os.path.exists(result_path + '/Images_Norm'):
                os.makedirs(result_path + '/Images_Norm')
            with open(result_path + '/Criteria.csv', 'w', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(['image_name', 'PSNR', 'SSIM', 'Time', 'Sigma_s', 'Sigma_r'])
                valid_SIDD(opt.dir_data + '/Test/SIDD', result_path, model, args, f_csv)

            print('===> Testing on SIDD Benchmark')
            result_path = opt.model_path + '/Results/SIDD Benchmark'
            if not os.path.exists(result_path + '/Images'):
                os.makedirs(result_path + '/Images')
            if not os.path.exists(result_path + '/Images_Norm'):
                os.makedirs(result_path + '/Images_Norm')
            test_SIDD(opt.dir_data + '/Test/SIDD', result_path, model, args)

            print('===> Testing on DND benchmark')
            result_path = opt.model_path + '/Results/DND Benchmark'
            if not os.path.exists(result_path + '/Images'):
                os.makedirs(result_path + '/Images')
            if not os.path.exists(result_path + '/Images_Norm'):
                os.makedirs(result_path + '/Images_Norm')
            test_DND(opt.dir_data + '/Test/DND', result_path, model, args)

        elif 'CBSD68' in opt.data_test:
            sigma_s = [np.array([0]), np.array([0.15]), np.array([0.3])]
            sigma_r = [np.array([15 / 255]), np.array([25 / 255]), np.array([50 / 255])]
            for data_test in opt.data_test:
                for sigma_s_ in sigma_s:
                    Avg_PSNR, Avg_SSIM = 0, 0
                    count = 0
                    for sigma_r_ in sigma_r:
                        print('===> Validation on {} with sigma_S={} and sigma_C={}'.format(data_test, sigma_s_,
                                                                                            sigma_r_))
                        result_path = opt.model_path + '/Results/{}/S{}C{}'.format(data_test, float(sigma_s_),
                                                                                           int(sigma_r_ * 255))
                        if not os.path.exists(result_path):
                            os.makedirs(result_path)
                        prior = np.append(sigma_s_, sigma_r_, axis=0)

                        with open(result_path + '/Criteria.csv', 'w', newline='') as f:
                            f_csv = csv.writer(f)
                            f_csv.writerow(['image_name', 'PSNR', 'Time', 'Sigma_s', 'Sigma_r'])
                            PSNR, SSIM, NoiP, Time = valid_PGNoise(opt.dir_data + 'Test/{}/HR'.format(data_test),
                                                                   result_path, model, prior, opt, f_csv)
                        Avg_PSNR += PSNR
                        Avg_SSIM += SSIM
                        count += 1
                    print('Avg PSNR / SSIM on {} with sigma_S={} = {:.2f} / {:.4f}'.
                          format(data_test, sigma_s_, Avg_PSNR / count, Avg_SSIM / count))

    elif opt.train == 'complexity':
        n_samples = 100
        torch.cuda.empty_cache()
        input = torch.FloatTensor(1, 3, 512, 512).to(opt.device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        Avg_Time = 0
        count = 0
        with torch.no_grad():
            with tqdm(total=n_samples) as pbar:
                for i in range(n_samples):
                    start.record()
                    if opt.method == 'Condformer':
                        input_patch = utils.random_cropping(input, 32, 8)
                        prior = model['LoNPE'](input_patch)
                        prior = prior.contiguous().view(8, 1, 2).median(dim=0)[0].detach()
                        model['Condformer'](input, prior)
                    else:
                        model(input)

                    end.record()
                    torch.cuda.synchronize()
                    Time = start.elapsed_time(end)
                    Avg_Time += Time
                    count += 1
                    pynvml.nvmlInit()
                    mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)) / 1024 ** 2
                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(TIME='{:.1f}ms'.format(Avg_Time / count),
                                     MEMO='{:.1f}MB'.format(mem))

                    torch.cuda.empty_cache()

        if opt.method == 'Condformer':
            for item in model:
                if 'Condformer' in item:
                    input = torch.FloatTensor(1, 3, 512, 512).to(opt.device)
                    prior = torch.FloatTensor(1, 2).to(opt.device)
                    FLOPs, Params = profile_origin(model[item].module if len(opt.GPUs) > 1 else model[item],
                                                   inputs=(input, prior,), verbose=False)
                elif 'LoNPE' in item:
                    input = torch.FloatTensor(8, 3, 32, 32).to(opt.device)
                    FLOPs, Params = profile_origin(model[item].module if len(opt.GPUs) > 1 else model[item],
                                                   inputs=(input,), verbose=False)
                else:
                    raise InterruptedError

                print('-------------{}-------------'.format(item))
                print('\tParam = {:.3f}K\n\tFLOPs = {:.3f}G on {}'.format(Params * 1e-3, FLOPs * 1e-9, input.shape))

if __name__ == '__main__':
    main()
