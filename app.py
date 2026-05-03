from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os
import uuid
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def resize_image(img, max_width=1200):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))
    return img


def compute_psnr_ssim(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))

    psnr = cv2.PSNR(img1, img2)

    def ssim_channel(a, b):
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        C1, C2 = 6.5025, 58.5225
        mu1 = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(b, (11, 11), 1.5)
        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(a*a, (11,11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(b*b, (11,11), 1.5) - mu2_sq
        sigma12   = cv2.GaussianBlur(a*b, (11,11), 1.5) - mu1_mu2
        num = (2*mu1_mu2 + C1)*(2*sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)
        return np.mean(num / den)

    ssim = np.mean([ssim_channel(img1[:,:,i], img2[:,:,i]) for i in range(3)])
    return round(psnr, 2), round(float(ssim), 4)


def draw_keypoints(img, keypoints):
    out = img.copy()
    out = cv2.drawKeypoints(out, keypoints[:500], None,
                            color=(0, 200, 255),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return out


def detect_and_match(img1, img2, detector_name):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if detector_name == 'SIFT':
        det = cv2.SIFT_create()
    elif detector_name == 'ORB':
        det = cv2.ORB_create(2000)
    else:
        det = cv2.AKAZE_create()

    t0 = time.time()
    kp1, des1 = det.detectAndCompute(gray1, None)
    kp2, des2 = det.detectAndCompute(gray2, None)
    elapsed = round(time.time() - t0, 3)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, None, 0, 0, elapsed

    if detector_name == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
    elif detector_name == 'AKAZE':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
    else:
        des1 = np.float32(des1)
        des2 = np.float32(des2)
        flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
        matches = flann.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    inlier_ratio = round(len(good) / max(len(matches), 1) * 100, 1)
    return kp1, kp2, len(good), inlier_ratio, elapsed


def stitch_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(resize_image(img))

    if len(images) < 2:
        return None, None, None, None, "Please upload at least 2 images."

    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, result = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        msgs = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough overlap. Try images with at least 30% overlap.",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Could not align images. Make sure they overlap.",
        }
        return None, None, None, None, msgs.get(status, "Stitching failed. Please try different images.")

    out_id = uuid.uuid4().hex
    result_path = os.path.join(UPLOAD_FOLDER, f"panorama_{out_id}.jpg")
    cv2.imwrite(result_path, result)

    img1, img2 = images[0], images[1]

    # Keypoint visualization
    kp1, kp2, good_matches, inlier_ratio, _ = detect_and_match(img1, img2, 'SIFT')
    kp_path = None
    if kp1 is not None:
        kp_img1 = draw_keypoints(img1, kp1)
        kp_img2 = draw_keypoints(img2, kp2)
        h1, w1 = kp_img1.shape[:2]
        h2, w2 = kp_img2.shape[:2]
        h = max(h1, h2)
        canvas = np.zeros((h, w1 + w2 + 10, 3), dtype=np.uint8)
        canvas[:h1, :w1] = kp_img1
        canvas[:h2, w1+10:w1+10+w2] = kp_img2
        kp_path = os.path.join(UPLOAD_FOLDER, f"keypoints_{out_id}.jpg")
        cv2.imwrite(kp_path, canvas)

    # PSNR / SSIM
    psnr, ssim = compute_psnr_ssim(img1, img2)

    # Detector comparison
    comparison = []
    for name in ['SIFT', 'ORB', 'AKAZE']:
        kp1_, kp2_, matches_, inlier_, t_ = detect_and_match(img1, img2, name)
        comparison.append({
            'name': name,
            'keypoints': len(kp1_) if kp1_ is not None else 0,
            'good_matches': matches_,
            'inlier_ratio': inlier_,
            'time_ms': int(t_ * 1000)
        })

    metrics = {
        'psnr': psnr,
        'ssim': ssim,
        'psnr_ok': psnr >= 30,
        'ssim_ok': ssim >= 0.85,
        'good_matches': good_matches,
        'inlier_ratio': inlier_ratio,
    }

    return result_path, kp_path, metrics, comparison, None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/stitch', methods=['POST'])
def stitch():
    files = request.files.getlist('images')
    if len(files) < 2:
        return render_template('index.html', error="Please upload at least 2 images.")

    saved_paths = []
    for f in files:
        if f.filename == '':
            continue
        path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{f.filename}")
        f.save(path)
        saved_paths.append(path)

    if len(saved_paths) < 2:
        return render_template('index.html', error="Please upload at least 2 valid image files.")

    result_path, kp_path, metrics, comparison, error = stitch_images(saved_paths)

    for p in saved_paths:
        try:
            os.remove(p)
        except:
            pass

    if error:
        return render_template('index.html', error=error)

    return render_template('index.html',
                           result=result_path,
                           kp_path=kp_path,
                           metrics=metrics,
                           comparison=comparison)


@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True, download_name="panorama.jpg")


@app.route('/view/<path:filename>')
def view_file(filename):
    return send_file(filename)


if __name__ == '__main__':
    print("\n✅ Panorama app is running!")
    print("👉 Open your browser and go to: http://127.0.0.1:5000\n")
    app.run(host='127.0.0.1', port=8080, debug=True)

