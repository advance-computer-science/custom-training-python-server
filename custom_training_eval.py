import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model_architechture import UNet 


IMAGES_DIR = "./test_dataset/images"
MASKS_DIR = "./test_dataset/masks"
WEIGHTS_FILE = "unet_lgg_weights.pth"

def evaluate_binary_segmentation(predicted_mask, true_mask, smooth=1e-6):
    pred = predicted_mask.view(-1).float()
    truth = true_mask.view(-1).float()
    
    intersection = (pred * truth).sum()
    total_pred = pred.sum()
    total_truth = truth.sum()
    
    dice = (2.0 * intersection + smooth) / (total_pred + total_truth + smooth)
    
    union = total_pred + total_truth - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return dice.item(), iou.item()


def run_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on: {device}")

    # load the model
    model = UNet(in_channels=3, out_channels=1) 
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    model.to(device)
    model.eval()

    # preprocessing
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    # define variables
    total_dice = 0.0
    total_iou = 0.0
    num_images = 0

    # iterate through all images in the directory
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.tif', '.png', '.jpg'))]
    
    print(f"Found {len(image_files)} images to evaluate.\n")

    with torch.no_grad():
        for filename in image_files:
            img_path = os.path.join(IMAGES_DIR, filename)
            
            name, ext = os.path.splitext(filename)
            mask_filename = f"{name}_mask{ext}" 
            mask_path = os.path.join(MASKS_DIR, mask_filename)

            if not os.path.exists(mask_path):
                print(f"Warning: No matching mask found for {filename} (Looked for {mask_filename}). Skipping.")
                continue

            raw_image = Image.open(img_path).convert("RGB")
            input_tensor = img_transform(raw_image).unsqueeze(0).to(device)

            # load ground truth mask (Convert to grayscale 'L')
            raw_mask = Image.open(mask_path).convert("L")
            truth_tensor = mask_transform(raw_mask).to(device)
            
            truth_mask = (truth_tensor > 0.5).float()

            # Run Inference
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            prediction_mask = (probs > 0.5).float()

            # Calculate Metrics
            dice, iou = evaluate_binary_segmentation(prediction_mask, truth_mask)
            
            total_dice += dice
            total_iou += iou
            num_images += 1
            
            print(f"File: {filename} | Dice: {dice:.4f} | IoU: {iou:.4f}")

    # Calculate and display final averages
    if num_images > 0:
        avg_dice = total_dice / num_images
        avg_iou = total_iou / num_images
        print("-" * 40)
        print("EVALUATION COMPLETE")
        print(f"Total Images Processed: {num_images}")
        print(f"Average Dice Coefficient: {avg_dice:.4f}")
        print(f"Average IoU Score:        {avg_iou:.4f}")
        print("-" * 40)
    else:
        print("No valid image/mask pairs were found to evaluate.")

if __name__ == "__main__":
    run_evaluation()