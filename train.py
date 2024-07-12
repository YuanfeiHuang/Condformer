import os, time, torch, h5py, cv2
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import utils as utils
from tqdm import tqdm
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
from scipy.io import loadmat, savemat
import data.common as common
from torch.cuda.amp import autocast
import base64
import pandas as pd


def train_SIDD(training_dataloader, optimizer, model, epoch, writer, args):
    torch.cuda.empty_cache()
    criterion_MAE = nn.L1Loss(reduction='sum').to(args.device)
    for item in model:
        model[item].train()

    with tqdm(total=len(training_dataloader)) as pbar:
        for iteration, (img_clean, img_noisy, img_sigma) in enumerate(training_dataloader):

            img_clean, img_noisy, img_sigma = Variable(img_clean).to(args.device), \
                                              Variable(img_noisy).to(args.device), \
                                              Variable(img_sigma).to(args.device)

            if args.lambda_NE == np.Inf:
                prior = Variable(img_sigma)
            elif args.lambda_NE > 0:
                # with autocast():
                optimizer['LoNPE'].zero_grad()
                img_noisy_patch = utils.random_cropping(img_noisy, patch_size=32, number=8)
                prior = model['LoNPE'](img_noisy_patch)
                img_sigma = Variable(img_sigma.repeat(8, 1), requires_grad=False)
                loss_NE = criterion_MAE(prior, img_sigma)
                # scaler.scale(args.lambda_NE * loss_NE).backward()
                # scaler.step(optimizer['LoNPE'])
                (args.lambda_NE*loss_NE).backward()
                optimizer['LoNPE'].step()
                # optimizer['LoNPE'].zero_grad(set_to_none=True)
                # scaler.update()
                prior = prior.contiguous().view(8, img_clean.shape[0], 2).median(dim=0)[0].detach()
            elif args.lambda_NE == 0:
                with autocast():
                    img_noisy_patch = utils.random_cropping(img_noisy, patch_size=32, number=8)
                    prior = model['LoNPE'](img_noisy_patch)
                    prior = prior.contiguous().view(8, img_clean.shape[0], 2).median(dim=0)[0].detach()
            elif args.lambda_NE < 0:
                prior = Variable(torch.zeros_like(img_sigma))

            if args.lambda_DN > 0:
                # with autocast():
                optimizer['Condformer'].zero_grad()
                img_denoised, embeddings = model['Condformer'](img_noisy, prior)
                loss_DN = criterion_MAE(img_denoised, img_clean)
                # scaler.scale(loss).backward()
                # scaler.step(optimizer['Condformer'])
                # optimizer['Condformer'].zero_grad(set_to_none=True)
                # scaler.update()
                (args.lambda_DN*loss_DN).backward()
                optimizer['Condformer'].step()

            time.sleep(0.01)
            pbar.update(1)
            pbar.set_postfix(Epoch=epoch,
                             LeaRate='{:.2e}-{:.2e}'.format(optimizer['LoNPE'].param_groups[0]['lr'],
                                                            optimizer['Condformer'].param_groups[0]['lr']),
                             Loss='DN{}*{:.2e}; NE{}*{:.2e}'
                             .format(args.lambda_DN, loss_DN if 'loss_DN' in locals().keys() else 0,
                                     args.lambda_NE, loss_NE if 'loss_NE' in locals().keys() else 0))

            niter = (epoch - 1) * len(training_dataloader) + iteration

            if (niter + 1) % 500 == 0:
                if 'loss_DN' in locals().keys():
                    writer.add_scalar('Train-Loss/loss_DN', loss_DN, niter)
                if 'loss_NE' in locals().keys():
                    writer.add_scalar('Train-Loss/loss_NE', loss_NE, niter)


def valid_SIDD(source_path, result_path, model, args, f_csv):
    if args.method == 'Condformer':
        for item in model:
            model[item].eval()
    else:
        model.eval()
    if args.use_matlab:
        import matlab.engine
        eng = matlab.engine.start_matlab()
    count = 0
    Avg_NoiP = 0
    Avg_PSNR = 0
    Avg_SSIM = 0
    Avg_Time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    factor = 8
    noisy = loadmat(os.path.join(source_path, 'ValidationNoisyBlocksSrgb.mat'))['ValidationNoisyBlocksSrgb']
    clean = loadmat(os.path.join(source_path, 'ValidationGtBlocksSrgb.mat'))['ValidationGtBlocksSrgb']
    n_im, n_block, h, w, c = noisy.shape
    # denoised = np.zeros((n_im, n_block, h, w, c), dtype=np.float32)

    torch.cuda.empty_cache()

    with torch.no_grad():
        with tqdm(total=n_im * (n_block)) as pbar:
            for i_img in range(0, n_im):
                prior = np.load(os.path.join(source_path + '/sigma_validation', 'Img{:02d}_SIGMA.npy'.format(i_img)))

                for i_block in range(0, n_block):
                    img_noisy, img_clean = noisy[i_img, i_block], clean[i_img, i_block]
                    img_noisy = common.set_channel(img_noisy, args.n_colors)
                    img_noisy = common.np2Tensor(img_noisy, args.value_range)

                    img_noisy = Variable(img_noisy[None]).to(args.device)
                    start.record()
                    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                    padh = H - h if h % factor != 0 else 0
                    padw = W - w if w % factor != 0 else 0
                    img_noisy_pad = F.pad(img_noisy, (0, padw, 0, padh), 'reflect')

                    prior_est = torch.from_numpy(np.zeros_like(prior)).float().view(1, 2).to(args.device)

                    if args.method == 'Condformer':
                        if args.lambda_NE == np.Inf:
                            prior_est = torch.from_numpy(prior).float().view(1, 2).to(args.device)
                        elif args.lambda_NE >= 0:
                            img_noisy_patch = utils.random_cropping(img_noisy_pad, 32, 8)
                            prior_est = model['LoNPE'].forward(img_noisy_patch)
                            prior_est = prior_est.contiguous().view(8, 1, 2).median(dim=0)[0].detach()
                        img_denoised, z_embeddings = model['Condformer'].forward(img_noisy_pad,
                                                                                 prior_est)
                    else:
                        img_denoised = model(img_noisy_pad)

                    img_denoised = img_denoised[:, :, :h, :w]
                    end.record()
                    torch.cuda.synchronize()
                    Time = start.elapsed_time(end)
                    Avg_Time += Time
                    count += 1

                    img_denoised = img_denoised.float().data[0].cpu().clamp(0, 1)
                    # denoised[i_img, i_block] = img_denoised.numpy().transpose(1, 2, 0)
                    prior_est = prior_est.data[0].cpu().numpy()
                    Avg_NoiP += -10 * np.log10(((prior - prior_est) ** 2).mean())

                    if args.use_matlab:
                        img_denoised_v = np.ascontiguousarray(img_denoised.numpy().transpose(1, 2, 0))
                        img_clean_v = np.ascontiguousarray(img_clean.astype(np.float32) / 255)
                        PSNR = eng.psnr(img_denoised_v, img_clean_v)
                        SSIM = eng.ssim(img_denoised_v, img_clean_v)
                    else:
                        img_clean = common.set_channel(img_clean, args.n_colors)
                        img_clean = common.np2Tensor(img_clean, args.value_range)
                        PSNR = utils.calc_PSNR(img_denoised, img_clean, args.value_range, shave=0, quantization=False)
                        # SSIM = utils.calc_SSIM(img_denoised, img_clean, args.value_range, shave=0, quantization=False)
                        SSIM = 0
                    Avg_PSNR += PSNR
                    Avg_SSIM += SSIM

                    if args.save_img:
                        img_noisy = img_noisy.data[0].cpu().clamp(0, 1)
                        MAX = torch.max(img_noisy.view(3, -1), dim=1)[0].view(3, 1, 1)
                        MIN = torch.min(img_noisy.view(3, -1), dim=1)[0].view(3, 1, 1)
                        img_denoised_show = utils.save_img((img_denoised - MIN) / (MAX - MIN), 3, 255)
                        img_denoised_show.save(result_path + '/Images_Norm/Image{:2d}_Block{:2d}_{}.png'
                                               .format(i_img + 1, i_block + 1, args.method))
                        img_denoised = utils.save_img(img_denoised, 3, 255)
                        img_denoised.save(result_path + '/Images/Image{:2d}_Block{:2d}_{}.png'
                                          .format(i_img + 1, i_block + 1, args.method))

                        img_clean_show = utils.save_img((img_clean - MIN) / (MAX - MIN), 3, 255)
                        img_clean_show.save(result_path + '/Images_Norm/Image{:2d}_Block{:2d}_Clean.png'
                                               .format(i_img + 1, i_block + 1))
                        img_clean = utils.save_img(img_clean, 3, 255)
                        img_clean.save(result_path + '/Images/Image{:2d}_Block{:2d}_Clean.png'
                                          .format(i_img + 1, i_block + 1))

                    if f_csv:
                        f_csv.writerow(['Image{:2d}_Block{:2d}'.format(i_img + 1, i_block + 1), PSNR, SSIM, Time,
                                        prior_est[0], prior_est[1]])

                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(Sigma_Diff='{:.2f}dB'.format(Avg_NoiP / count) if args.method == 'Condformer' else '',
                                     Sigma_EST='S{:.3f}-C{:.3f}'.format(prior_est[0],
                                                                        prior_est[1]) if args.method == 'Condformer' else '',
                                     Sigma_GT='S{:.3f}-C{:.3f}'.format(prior[0],
                                                                       prior[1]) if args.method == 'Condformer' else '',
                                     PSNR='{:.2f}dB'.format(Avg_PSNR / count),
                                     SSIM='{:.4f}'.format(Avg_SSIM / count),
                                     count=count,
                                     TIME='{:.1f}ms'.format(Avg_Time / count))

        # savemat(result_path + '/ValidSrgb.mat', {'DenoisedBlocksSrgb': denoised})

    if f_csv:
        f_csv.writerow(['Average', Avg_PSNR / count, Avg_SSIM / count, Avg_Time / count, '', ''])

    torch.cuda.empty_cache()
    return Avg_PSNR / count, Avg_SSIM / count, Avg_NoiP / count, Avg_Time / count


def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


def test_SIDD(source_path, result_path, model, args):
    if isinstance(model, dict):
        for item in model:
            model[item].eval()
    elif model is not None:
        model.eval()
    count = 0
    Avg_NoiP = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    factor = 8
    Avg_Time = 0
    torch.cuda.empty_cache()
    noisy = loadmat(os.path.join(source_path, 'BenchmarkNoisyBlocksSrgb.mat'))['BenchmarkNoisyBlocksSrgb']
    n_im, n_block, h, w, c = noisy.shape
    # denoised = np.zeros((n_im, n_block, h, w, c), dtype=np.uint8)
    output_blocks_base64string = []

    with torch.no_grad():
        with tqdm(total=n_im * n_block) as pbar:
            for i_img in range(0, n_im):
                prior = np.load(os.path.join(source_path + '/sigma_benchmark', 'Img{:02d}_SIGMA.npy'.format(i_img)))
                for i_block in range(0, n_block):
                    img_noisy = noisy[i_img, i_block]
                    img_noisy = common.set_channel(img_noisy, 3)
                    img_noisy = common.np2Tensor(img_noisy, 255)

                    img_noisy = Variable(img_noisy[None]).to(args.device)

                    start.record()
                    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                    padh = H - h if h % factor != 0 else 0
                    padw = W - w if w % factor != 0 else 0
                    img_noisy_pad = F.pad(img_noisy, (0, padw, 0, padh), 'reflect')

                    prior_est = torch.from_numpy(np.zeros_like(prior)).float().view(1, 2).to(args.device)

                    if args.method == 'Condformer':
                        if args.lambda_NE == np.Inf:
                            prior_est = torch.from_numpy(prior).float().view(1, 2).to(args.device)
                        elif args.lambda_NE >= 0:
                            img_noisy_patch = utils.random_cropping(img_noisy_pad, 32, 8)
                            prior_est = model['LoNPE'].forward(img_noisy_patch)
                            prior_est = prior_est.contiguous().view(8, 1, 2).median(dim=0)[0].detach()

                        img_denoised, z_embeddings = model['Condformer'].forward(img_noisy_pad,
                                                                                 prior_est)
                    else:
                        img_denoised = model(img_noisy_pad)

                    # Unpad images to original dimensions
                    img_denoised = img_denoised[:, :, :h, :w]
                    end.record()
                    torch.cuda.synchronize()
                    Time = start.elapsed_time(end)
                    Avg_Time += Time
                    count += 1

                    img_denoised = img_denoised.float().data[0].cpu().clamp(0, 1)
                    prior_est = prior_est.data[0].cpu().numpy()

                    Avg_NoiP += -10 * np.log10(((prior - prior_est) ** 2).mean())

                    denoised = img_denoised.mul(255).round().numpy().astype(np.uint8).transpose(1, 2, 0)
                    out_block_base64string = array_to_base64string(denoised)
                    output_blocks_base64string.append(out_block_base64string)

                    if args.save_img:
                        img_noisy = img_noisy.data[0].cpu().clamp(0, 1)
                        MAX = img_noisy.max()
                        MIN = img_noisy.min()
                        img_denoised_show = utils.save_img((img_denoised - MIN) / (MAX - MIN), 3, 255)
                        img_denoised_show.save(result_path + '/Images_Norm/Image{:2d}_Block{:2d}_{}.png'
                                               .format(i_img + 1, i_block + 1, args.method))
                        img_denoised = utils.save_img(img_denoised, 3, 255)
                        img_denoised.save(result_path + '/Images/Image{:2d}_Block{:2d}_{}.png'
                                          .format(i_img + 1, i_block + 1, args.method))

                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(Sigma_Diff='{:.2f}dB'.format(Avg_NoiP / count),
                                     Sigma_EST='S{:.3f}-C{:.3f}'.format(prior_est[0], prior_est[1]),
                                     Sigma_GT='S{:.3f}-C{:.3f}'.format(prior[0], prior[1]),
                                     TIME='{:.1f}ms'.format(Avg_Time / count),
                                     )

    # Save outputs to .csv file.
    output_file = result_path + '/SubmitSrgb.csv'
    print(f'Saving outputs to {output_file}')
    output_df = pd.DataFrame()
    n_blocks = len(output_blocks_base64string)
    print(f'Number of blocks = {n_blocks}')
    output_df['ID'] = np.arange(n_blocks)
    output_df['BLOCK'] = output_blocks_base64string

    output_df.to_csv(output_file, index=False)

    # TODO: Submit the output file SubmitSrgb.csv at
    # kaggle.com/competitions/sidd-benchmark-srgb-psnr
    print('TODO: Submit the output file SubmitSrgb.csv at')
    print('kaggle.com/competitions/sidd-benchmark-srgb-psnr')
    print('Done.')

    torch.cuda.empty_cache()
    return Avg_NoiP / count, Avg_Time / count


def test_DND(source_path, result_path, model, args):
    if isinstance(model, dict):
        for item in model:
            model[item].eval()
    elif model is not None:
        model.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    factor = 8
    Avg_Time = 0
    torch.cuda.empty_cache()

    infos = h5py.File(os.path.join(source_path, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']

    path_img = os.listdir(source_path + '/images_srgb')
    path_img.sort()
    n_im, n_block, h, w, c = 50, 20, 512, 512, 3
    denoised = np.zeros((n_block, ), dtype=object)
    Avg_NoiP = 0
    count = 0

    with torch.no_grad():
        with tqdm(total=n_im * n_block) as pbar:
            for i_img in range(0, n_im):
                img = h5py.File(os.path.join(source_path + '/images_srgb', path_img[i_img]), 'r')
                img = np.float32(np.array(img['InoisySRGB']).T)
                ref = bb[0][i_img]
                boxes = np.array(info[ref]).T

                prior = np.load(os.path.join(source_path + '/sigma', 'Img{:02d}_SIGMA.npy'.format(i_img)))

                for i_block in range(0, n_block):
                    idx = [int(boxes[i_block, 0] - 1), int(boxes[i_block, 2]), int(boxes[i_block, 1] - 1),
                           int(boxes[i_block, 3])]
                    img_noisy = np.float32(img[idx[0]:idx[1], idx[2]:idx[3], :]) * 255
                    img_noisy = common.np2Tensor(img_noisy, 255)
                    img_noisy = Variable(img_noisy[None]).to(args.device)

                    start.record()
                    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                    padh = H - h if h % factor != 0 else 0
                    padw = W - w if w % factor != 0 else 0
                    img_noisy_pad = F.pad(img_noisy, (0, padw, 0, padh), 'reflect')

                    prior_est = torch.from_numpy(np.zeros_like(prior)).float().view(1, 2).to(args.device)

                    if args.method == 'Condformer':
                        if args.lambda_NE == np.Inf:
                            prior_est = torch.from_numpy(prior).float().view(1, 2).to(args.device)
                        elif args.lambda_NE >= 0:
                            img_noisy_patch = utils.random_cropping(img_noisy_pad, 32, 8)
                            prior_est = model['LoNPE'].forward(img_noisy_patch)
                            prior_est = prior_est.contiguous().view(8, 1, 2).median(dim=0)[0].detach()

                        img_denoised, z_embeddings = model['Condformer'].forward(img_noisy_pad,
                                                                                 prior_est)
                    else:
                        img_denoised = model(img_noisy_pad)

                    # Unpad images to original dimensions
                    img_denoised = img_denoised[:, :, :h, :w]
                    end.record()
                    torch.cuda.synchronize()
                    Time = start.elapsed_time(end)
                    Avg_Time += Time
                    count += 1

                    prior_est = prior_est.data[0].cpu().numpy()

                    Avg_NoiP += -10 * np.log10(((prior - prior_est) ** 2).mean())

                    img_denoised = img_denoised.float().data[0].cpu().clamp(0, 1)

                    denoised[i_block] = np.float32(img_denoised).transpose(1, 2, 0)

                    if args.save_img:
                        img_noisy = img_noisy.data[0].cpu().clamp(0, 1)
                        MAX = img_noisy.max()
                        MIN = img_noisy.min()
                        img_denoised_show = utils.save_img((img_denoised - MIN) / (MAX - MIN), 3, 255)
                        img_denoised_show.save(result_path + '/Images_Norm/Image{:2d}_Block{:2d}_{}.png'
                                               .format(i_img + 1, i_block + 1, args.method))
                        img_denoised = utils.save_img(img_denoised, 3, 255)
                        img_denoised.save(result_path + '/Images/Image{:2d}_Block{:2d}_{}.png'
                                          .format(i_img + 1, i_block + 1, args.method))

                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(Sigma_Diff='{:.2f}dB'.format(Avg_NoiP / count),
                                     Sigma_EST='S{:.3f}-C{:.3f}'.format(prior_est[0], prior_est[1]),
                                     Sigma_GT='S{:.3f}-C{:.3f}'.format(prior[0], prior[1]),
                                     TIME='{:.1f}ms'.format(Avg_Time / count),
                                     )

                if not os.path.exists(result_path + '/MAT'):
                    os.makedirs(result_path + '/MAT')
                savemat(
                    os.path.join(result_path + '/MAT', '{:04d}.mat'.format(i_img + 1)),
                    {"Idenoised": denoised,
                     "israw": False,
                     "eval_version": "1.0"},
                )
    torch.cuda.empty_cache()
    return Avg_NoiP / count, Avg_Time / count


def train_DF2K(training_dataloader, optimizer, model, epoch, writer, args):
    scaler = torch.cuda.amp.GradScaler()
    criterion_MAE = nn.L1Loss(reduction='sum').to(args.device)
    for item in model:
        model[item].train()
    with tqdm(total=len(training_dataloader)) as pbar:
        for iteration, (img_clean) in enumerate(training_dataloader):
            img_clean = Variable(img_clean).to(args.device)

            noise_std_s = np.zeros((2, 1))
            noise_std_s = np.append(noise_std_s, np.array(np.random.uniform(0, 0.3, size=(args.batch_size - 2, 1))), axis=0)
            noise_std_c = np.array(np.random.uniform(0, 0.2, size=(args.batch_size, 1)))
            img_sigma = torch.from_numpy(np.append(noise_std_s, noise_std_c, axis=1)).float().to(args.device)
            img_noisy = img_clean + \
                        torch.normal(0, torch.sqrt(
                            (img_sigma[:, 0:1, None, None] ** 2) * img_clean + img_sigma[:, 1:2, None, None] ** 2))
            img_noisy = Variable(img_noisy).to(args.device)

            if args.lambda_NE == np.Inf:
                prior = Variable(img_sigma)
            elif args.lambda_NE > 0:
                # with autocast():
                img_noisy_patch = utils.random_cropping(img_noisy, patch_size=32, number=8)
                prior = model['LoNPE'](img_noisy_patch)
                loss_NE = criterion_MAE(prior, Variable(img_sigma.repeat(8, 1), requires_grad=False))
                (args.lambda_NE * loss_NE).backward()
                optimizer['LoNPE'].step()
                # scaler.scale(args.lambda_NE * loss_NE).backward()
                # # scaler.unscale_(optimizer['LoNPE'])
                # # torch.nn.utils.clip_grad_value_(model['LoNPE'].parameters(), 0.1)
                # scaler.step(optimizer['LoNPE'])
                optimizer['LoNPE'].zero_grad(set_to_none=True)
                # scaler.update()
                prior = prior.contiguous().view(8, img_clean.shape[0], 2).median(dim=0)[0].detach()
            elif args.lambda_NE == 0:
                # with autocast():
                img_noisy_patch = utils.random_cropping(img_noisy, patch_size=32, number=8)
                prior = model['LoNPE'](img_noisy_patch)
                prior = prior.contiguous().view(8, img_clean.shape[0], 2).median(dim=0)[0].detach()
            elif args.lambda_NE < 0:
                prior = Variable(torch.zeros_like(img_sigma))

            if args.lambda_DN > 0:
                # with autocast():
                img_denoised, embeddings = model['Condformer'](img_noisy, prior)
                loss_DN = criterion_MAE(img_denoised, img_clean)
                (args.lambda_DN * loss_DN).backward()
                # scaler.scale(loss).backward()
                #                 scaler.unscale_(optimizer['Condformer'])
                #                 torch.nn.utils.clip_grad_value_(model['Condformer'].parameters(), 0.01)
                # scaler.step(optimizer['Condformer'])
                optimizer['Condformer'].step()
                optimizer['Condformer'].zero_grad(set_to_none=True)
                # scaler.update()

            time.sleep(0.01)
            pbar.update(1)
            pbar.set_postfix(Epoch=epoch,
                             LeaRate='{:.2e}-{:.2e}'.format(optimizer['LoNPE'].param_groups[0]['lr'],
                                                            optimizer['Condformer'].param_groups[0]['lr']),
                             Loss='DN{}*{:.2e}; NE{}*{:.2e}'
                             .format(args.lambda_DN, loss_DN if 'loss_DN' in locals().keys() else 0,
                                     args.lambda_NE, loss_NE if 'loss_NE' in locals().keys() else 0))

            niter = (epoch - 1) * len(training_dataloader) + iteration

            if (niter + 1) % 500 == 0:
                if 'loss_DN' in locals().keys():
                    writer.add_scalar('Train-Loss/loss_DN', loss_DN, niter)
                if 'loss_NE' in locals().keys():
                    writer.add_scalar('Train-Loss/loss_NE', loss_NE, niter)


def valid_PGNoise(source_path, result_path, model, prior, args, f_csv):
    if isinstance(model, dict):
        for item in model:
            model[item].eval()
    elif model is not None:
        model.eval()
    if args.use_matlab:
        import matlab.engine
        eng = matlab.engine.start_matlab()
    count = 0
    Avg_PSNR = 0
    Avg_SSIM = 0
    Avg_NoiP = 0
    Avg_Time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    factor = 8
    filename = os.listdir(source_path)

    torch.cuda.empty_cache()

    with torch.no_grad():
        with tqdm(total=len(filename)) as pbar:
            for img_name in filename:
                img_clean = cv2.cvtColor(cv2.imread(os.path.join(source_path, img_name)), cv2.COLOR_BGR2RGB)
                img_name, ext = os.path.splitext(img_name)
                img_clean = common.set_channel(img_clean, args.n_colors)
                img_clean = common.np2Tensor(img_clean, args.value_range)

                img_noisy = img_clean + torch.normal(0, torch.sqrt((prior[0] ** 2) * img_clean + prior[1] ** 2))
                img_noisy = Variable(img_noisy[None]).to(args.device)

                c, h, w = img_clean.shape
                start.record()
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                img_noisy_pad = F.pad(img_noisy, (0, padw, 0, padh), 'reflect')
                prior_est = torch.from_numpy(np.zeros_like(prior)).float().view(1, 2).to(args.device)

                if args.method == 'Condformer':
                    if args.lambda_NE == np.Inf:
                        prior_est = torch.from_numpy(prior).float().view(1, 2).to(args.device)
                    elif args.lambda_NE >= 0:
                        img_noisy_patch = utils.random_cropping(img_noisy_pad, 32, 8)
                        prior_est = model['LoNPE'].forward(img_noisy_patch)
                        prior_est = prior_est.contiguous().view(8, 1, 2).median(dim=0)[0].detach()

                    img_denoised, z_embeddings = model['Condformer'].forward(img_noisy_pad,
                                                                             prior_est)
                else:
                    img_denoised = model(img_noisy_pad)

                # Unpad images to original dimensions
                img_denoised = img_denoised[:, :, :h, :w]
                end.record()
                torch.cuda.synchronize()
                Time = start.elapsed_time(end)
                Avg_Time += Time
                count += 1

                img_denoised = img_denoised.float().data[0].cpu().clamp(0, 1)
                prior_est = prior_est.data[0].cpu().numpy()
                if args.use_matlab:
                    img_denoised = utils.quantize(img_denoised, rgb_range=255)
                    img_denoised_v = np.ascontiguousarray(img_denoised.numpy().transpose(1, 2, 0))
                    img_clean_v = np.ascontiguousarray(img_clean.numpy().astype(np.float32).transpose(1, 2, 0))
                    PSNR = eng.psnr(img_denoised_v, img_clean_v)
                    SSIM = eng.ssim(img_denoised_v, img_clean_v)
                else:
                    PSNR = utils.calc_PSNR(img_denoised, img_clean, args.value_range, shave=0, quantization=False)
                    SSIM = 0
                    # SSIM = utils.calc_SSIM(img_denoised, img_clean, args.value_range, shave=0, quantization=False)

                Avg_PSNR += PSNR
                Avg_SSIM += SSIM
                Avg_NoiP += -10 * np.log10(((prior - prior_est) ** 2).mean())

                if args.save_img:
                    img_denoised = utils.save_img(img_denoised, 3, 255)
                    img_denoised.save(result_path + '/{}.png'.format(img_name))

                if f_csv:
                    f_csv.writerow([img_name, PSNR, SSIM, Time, prior_est[0], prior_est[1]])

                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(PSNR='{:.2f}dB'.format(Avg_PSNR / count),
                                 SSIM='{:.4f}'.format(Avg_SSIM / count),
                                 Sigma_Diff='{:.2f}dB'.format(Avg_NoiP / count),
                                 Sigma_EST='S{:.3f}-C{:.3f}'.format(prior_est[0], prior_est[1]),
                                 Sigma_GT='S{:.3f}-C{:.3f}'.format(prior[0], prior[1]),
                                 TIME='{:.1f}ms'.format(Avg_Time / count),
                                 )

    if f_csv:
        f_csv.writerow(['Average', Avg_PSNR / count, Avg_SSIM / count, Avg_Time / count, '', ''])

    torch.cuda.empty_cache()

    return Avg_PSNR / count, Avg_SSIM / count, Avg_NoiP / count, Avg_Time / count

