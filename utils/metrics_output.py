import re


def parse_metrics_output(metrics_output):
    """
    Parse the output of metrics.py to extract key metrics such as PSNR/SSIM/LPIPS
    """
    metrics = {}

    psnr_match = re.search(r'PSNR:\s*([\d\.]+)', metrics_output)
    if psnr_match:
        metrics['PSNR'] = float(psnr_match.group(1))

    ssim_match = re.search(r'SSIM:\s*([\d\.]+)', metrics_output)
    if ssim_match:
        metrics['SSIM'] = float(ssim_match.group(1))

    lpips_match = re.search(r'LPIPS:\s*([\d\.]+)', metrics_output)
    if lpips_match:
        metrics['LPIPS'] = float(lpips_match.group(1))
    return metrics