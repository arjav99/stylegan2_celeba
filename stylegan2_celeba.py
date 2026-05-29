import torch
import torchvision
import matplotlib.pyplot as plt
import time
import os
import imageio.v2 as imageio
import copy

# Hyperparameters
BATCH_SIZE = 16
NUM_WORKERS = 2
LR = 0.0002
EPOCHS = 50
LAMBDA_R1 = 1
LAMBDA_PL = 0.5
MIXING_PROB = 0.9
R1_INTERVAL = 16
PL_INTERVAL = 4 
N_CRITIC = 1 # Rule of thumb: Set to 1 for modern StyleGAN architectures
IMG_SIZE = 64
SUB_SET_SIZE = 0.5
LATENT_DIM = 256
checkpoint_path = "checkpoint.pth"

class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.skip = EqualizedConv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        self.relu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        res = self.skip(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x + res

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initial projection
        self.from_rgb = torch.nn.Sequential(
            EqualizedConv2d(3, 64, kernel_size=1),
            torch.nn.LeakyReLU(0.2)
        )
        
        # Downsampling blocks
        self.block1 = DiscriminatorBlock(64, 128)   # 64x64 -> 32x32
        self.block2 = DiscriminatorBlock(128, 256)  # 32x32 -> 16x16
        self.block3 = DiscriminatorBlock(256, 512)  # 16x16 -> 8x8
        self.block4 = DiscriminatorBlock(512, 512)  # 8x8 -> 4x4

        self.final_conv = EqualizedConv2d(512 + 1, 512, 3, stride=1, padding=1) # +1 for Minibatch StdDev
        self.relu = torch.nn.LeakyReLU(0.2)
        
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            EqualizedLinear(512 * 4 * 4, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Minibatch Standard Deviation
        b, c, h, w = x.shape
        std = torch.std(x, dim=0, keepdim=True).mean()
        std = std.expand(b, 1, h, w)
        x = torch.cat([x, std], dim=1)
        
        x = self.relu(self.final_conv(x))
        return self.fc(x)


class EqualizedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.scale = (2 / in_features) ** 0.5 

    def forward(self, x):
        return torch.nn.functional.linear(x, self.scale * self.weight, self.bias)

class EqualizedConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        
        # Initialize weights from N(0, 1)
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        
        # He/Kaiming constant factor
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = (2 / fan_in) ** 0.5

    def forward(self, x):
        scaled_weight = self.weight * self.scale
        return torch.nn.functional.conv2d(
            x, scaled_weight, self.bias, stride=self.stride, padding=self.padding
        )

class PixelNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
    

class MappingNetwork(torch.nn.Module):
    def __init__(self, latent_dim=256, w_dim=256):
        super().__init__()
        # Removed self.num_layers since broadcasting shouldn't happen here
        
        layers = [PixelNorm()]
        in_dim = latent_dim
        for i in range(8):
            layers.append(EqualizedLinear(in_dim, w_dim))
            layers.append(torch.nn.LeakyReLU(0.2))
            in_dim = w_dim
            
        self.fc_block = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        # Fix 1: Just return the raw 256-dimensional w vector
        return self.fc_block(x)


class NoiseInjection(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(channels)) 

    def forward(self, x):
        noise = torch.randn_like(x)
        return x + self.weight.view(1, -1, 1, 1) * noise


class ModulatedConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, padding=0, demodulate=True, eps=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.demodulate = demodulate
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        self.scale = (2 / (in_channels * kernel_size * kernel_size)) ** 0.5

        self.style_fc = torch.nn.Linear(style_dim, in_channels)
        torch.nn.init.normal_(self.style_fc.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.style_fc.bias)

    def forward(self, x, style):
        B, Cin, H, W = x.shape
        
        # 1. Project latent w to style space
        style = self.style_fc(style)          
        
        # 2. CRITICAL: Add an affine bias of 1.0 so style scales around 1.0 instead of 0.0
        # This prevents weights from multiplying by 0 or flipping signs negatively.
        style = (style + 1.0).view(B, 1, Cin, 1, 1)   

        # 3. Apply equalized learning rate scale and modulate
        weight = self.weight[None] * self.scale * style   

        # 4. Demodulate to restore unit variance across the channel dimensions
        if self.demodulate:
            # Sum over Cin, Kh, Kw (dims 2, 3, 4)
            demod = torch.rsqrt((weight ** 2).sum(dim=[2, 3, 4]) + self.eps)  
            weight = weight * demod.view(B, self.out_channels, 1, 1, 1)

        # 5. Grouped convolution trick for batch processing
        x = x.reshape(1, B * Cin, H, W)
        weight = weight.reshape(B * self.out_channels, Cin, self.kernel_size, self.kernel_size)

        out = torch.nn.functional.conv2d(x, weight, padding=self.padding, groups=B)
        
        # 6. Reshape back to standard batch form and add bias
        out = out.reshape(B, self.out_channels, out.shape[2], out.shape[3])
        return out + self.bias.view(1, -1, 1, 1)


class SynthesisNetwork(torch.nn.Module):
    def __init__(self, w_dim=256, use_noise=True):
        super().__init__()
        self.use_noise = use_noise
        self.const_input = torch.nn.Parameter(torch.randn(1, 256, 4, 4)) 

        # Block 1
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1_1 = ModulatedConv2d(256, 512, 3, w_dim, padding=1)
        self.noise1_1 = NoiseInjection(512)
        self.relu1_1 = torch.nn.LeakyReLU(0.2)
        self.conv1_2 = ModulatedConv2d(512, 512, 3, w_dim, padding=1)
        self.noise1_2 = NoiseInjection(512)
        self.relu1_2 = torch.nn.LeakyReLU(0.2)
        self.to_rgb1 = ModulatedConv2d(512, 3, 1, w_dim, padding=0, demodulate=False) 

        # Block 2
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2_1 = ModulatedConv2d(512, 256, 3, w_dim, padding=1)
        self.noise2_1 = NoiseInjection(256)
        self.relu2_1 = torch.nn.LeakyReLU(0.2)
        self.conv2_2 = ModulatedConv2d(256, 256, 3, w_dim, padding=1)
        self.noise2_2 = NoiseInjection(256)
        self.relu2_2 = torch.nn.LeakyReLU(0.2)
        self.to_rgb2 = ModulatedConv2d(256, 3, 1, w_dim, padding=0, demodulate=False)

        # Block 3
        self.upsample3 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3_1 = ModulatedConv2d(256, 128, 3, w_dim, padding=1)
        self.noise3_1 = NoiseInjection(128)
        self.relu3_1 = torch.nn.LeakyReLU(0.2)
        self.conv3_2 = ModulatedConv2d(128, 128, 3, w_dim, padding=1)
        self.noise3_2 = NoiseInjection(128)
        self.relu3_2 = torch.nn.LeakyReLU(0.2)
        self.to_rgb3 = ModulatedConv2d(128, 3, 1, w_dim, padding=0, demodulate=False)

        # Block 4
        self.upsample4 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4_1 = ModulatedConv2d(128, 64, 3, w_dim, padding=1)
        self.noise4_1 = NoiseInjection(64)
        self.relu4_1 = torch.nn.LeakyReLU(0.2)
        self.conv4_2 = ModulatedConv2d(64, 64, 3, w_dim, padding=1)
        self.noise4_2 = NoiseInjection(64)
        self.relu4_2 = torch.nn.LeakyReLU(0.2)
        self.to_rgb4 = ModulatedConv2d(64, 3, 1, w_dim, padding=0, demodulate=False)

        self.tanh = torch.nn.Tanh()

    def forward(self, w_list):
        batch_size = w_list[0].shape[0]
        x = self.const_input.expand(batch_size, -1, -1, -1)  

        # Block 1 (Uses styles 0, 1, 2)
        x = self.upsample1(x)
        x = self.conv1_1(x, w_list[0])
        if self.use_noise: x = self.noise1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x, w_list[1])
        if self.use_noise: x = self.noise1_2(x)
        x = self.relu1_2(x)
        rgb = self.to_rgb1(x, w_list[2]) 

        # Block 2 (Uses styles 3, 4, 5)
        x = self.upsample2(x)
        x = self.conv2_1(x, w_list[3])
        if self.use_noise: x = self.noise2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x, w_list[4])
        if self.use_noise: x = self.noise2_2(x)
        x = self.relu2_2(x)
        rgb = self.to_rgb2(x, w_list[5]) + torch.nn.functional.interpolate(rgb, scale_factor=2)

        # Block 3 (Uses styles 6, 7, 8)
        x = self.upsample3(x)
        x = self.conv3_1(x, w_list[6])
        if self.use_noise: x = self.noise3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x, w_list[7])
        if self.use_noise: x = self.noise3_2(x)
        x = self.relu3_2(x)
        rgb = self.to_rgb3(x, w_list[8]) + torch.nn.functional.interpolate(rgb, scale_factor=2)

        # Block 4 (Uses styles 9, 10, 11)
        x = self.upsample4(x)
        x = self.conv4_1(x, w_list[9])
        if self.use_noise: x = self.noise4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x, w_list[10])
        if self.use_noise: x = self.noise4_2(x)
        x = self.relu4_2(x)
        rgb = self.to_rgb4(x, w_list[11]) + torch.nn.functional.interpolate(rgb, scale_factor=2)



        return self.tanh(rgb)


class PathLengthRegularizer(torch.nn.Module):
    def __init__(self, decay=0.99):
        super().__init__()
        self.decay = decay
        self.register_buffer("path_length_mean", torch.tensor(0.0))
    
    def forward(self, w_list, fake_images):
        batch_size = fake_images.shape[0]
        

        noise = torch.randn_like(fake_images) / torch.sqrt(
            torch.tensor(fake_images.numel() / batch_size, device=fake_images.device, dtype=fake_images.dtype)
        )
        
        loss_sum = (fake_images * noise).sum()
        
        grads_list = torch.autograd.grad(
            outputs=loss_sum,
            inputs=w_list,
            create_graph=True,
            retain_graph=True,
            allow_unused=True  
        )
        
        all_grads = torch.cat([g.reshape(batch_size, -1) for g in grads_list if g is not None], dim=1)
        path_lengths = torch.sqrt((all_grads ** 2).sum(dim=1) + 1e-8)
        
        path_length_mean = self.path_length_mean.lerp(
            path_lengths.mean(), 
            1 - self.decay
        )
        self.path_length_mean = path_length_mean.detach()
        
        pl_loss = ((path_lengths - path_length_mean) ** 2).mean()
        return pl_loss, path_length_mean.detach()


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        """
        Maintains a moving average of model parameters.
        Args:
            model (torch.nn.Module): The live network to track.
            decay (float): The smoothing factor (usually 0.999 for GANs).
        """
        self.decay = decay
        # Create a deep copy of the model to hold the averaged weights
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        
        # Disable gradient tracking for the EMA weights to save memory
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        """Updates the EMA weights using your in-place linear interpolation."""
        # θ_ema = θ_ema + (1 - decay) * (θ_model - θ_ema)
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.lerp_(model_param, 1.0 - self.decay)

        # Buffers (like running means in batch norm, if any) are copied directly
        for ema_buffer, model_buffer in zip(self.ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(model_buffer)


def logistic_loss_discriminator(pred_real, pred_fake):
    """
    Standard non-saturating logistic loss for the discriminator.
    """
    loss_real = torch.nn.functional.softplus(-pred_real)
    loss_fake = torch.nn.functional.softplus(pred_fake)
    return (loss_real + loss_fake).mean()


def logistic_loss_generator(pred_fake):
    """
    Standard non-saturating logistic loss for the generator.
    """
    return torch.nn.functional.softplus(-pred_fake).mean()


def compute_r1_penalty(discriminator, real_images, lambda_r1=10.0):
    """
    R1 Regularization: Penalizes gradients on REAL images only.
    """
    # Enforce gradient tracking on real images for this calculation
    real_images_grad = real_images.detach().requires_grad_(True)
    pred_real = discriminator(real_images_grad)
    
    # Calculate gradients of outputs w.r.t inputs
    grads = torch.autograd.grad(
        outputs=pred_real.sum(),
        inputs=real_images_grad,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # L2 norm squared of the gradients
    r1_penalty = grads.square().sum(dim=[1, 2, 3]).mean()
    
    return r1_penalty * (lambda_r1 * 0.5)


def apply_style_mixing(mapping_network, batch_size, latent_dim, device, mixing_prob=0.9):
    z1 = torch.randn(batch_size, latent_dim, device=device)
    w1 = mapping_network(z1) # Shape: [B, 256]

    if torch.rand(1).item() < mixing_prob:
        z2 = torch.randn(batch_size, latent_dim, device=device)
        w2 = mapping_network(z2) # Shape: [B, 256]

        # Pick a cross-over point out of the 12 total layers
        mixing_layer = torch.randint(1, 12, (1,)).item()

        # Fix 2: Blend the two styles structurally without redundant allocation
        w_list = [w1 for _ in range(mixing_layer)] + [w2 for _ in range(12 - mixing_layer)]
    else:
        w_list = [w1 for _ in range(12)]

    return w_list


def train_step(dataloader, discriminator, mapping_network, synthesis_network, 
               optim_discriminator, optim_mapping, optim_synthesis, latent_dim, 
               mapping_ema, synthesis_ema, device, n_critics, pl_regularizer, 
               lambda_r1=10.0, lambda_pl=2, mixing_prob=0.9, r1_interval=16, 
               pl_interval=16, global_step=0):
    
    total_disc_loss, total_gen_loss, total_r1_loss, total_pl_loss = 0.0, 0.0, 0.0, 0.0
    r1_count, pl_count = 0, 0
    
    discriminator.train()
    mapping_network.train()
    synthesis_network.train()
    
    step = global_step
    
    for real_images, _ in dataloader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device, non_blocking=True)
        
        ''' 1. Train Discriminator '''
        for _ in range(n_critics):
            # Generate fake images
            w_list = apply_style_mixing(mapping_network, batch_size, latent_dim, device, mixing_prob)
            w_list_det = [w.detach() for w in w_list] if isinstance(w_list, list) else w_list.detach()
            fake_images = synthesis_network(w_list_det).detach()

            # Enable requires_grad on real images UP FRONT if it's an R1 step ---
            if step % r1_interval == 0:
                real_images.requires_grad_(True)
            
            # Standard Logistic Forward Pass
            pred_real = discriminator(real_images)
            pred_fake = discriminator(fake_images)
            loss_disc = logistic_loss_discriminator(pred_real, pred_fake)
            
            # Lazy R1 Regularization
            r1_loss = torch.tensor(0.0, device=device)
            if step % r1_interval == 0:
                # --- FIX: Pass the active tensor directly ---
                grads = torch.autograd.grad(
                    outputs=pred_real.sum(),
                    inputs=real_images,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                r1_penalty = grads.square().sum(dim=[1, 2, 3]).mean()
                r1_loss = r1_penalty * (lambda_r1 * 0.5)
                
                total_r1_loss += r1_loss.item()
                r1_count += 1
            
            # Total D Loss (scaled for lazy regularization execution)
            total_loss_disc = loss_disc + (r1_loss * r1_interval)
            
            optim_discriminator.zero_grad(set_to_none=True)
            total_loss_disc.backward()
            optim_discriminator.step()
            
            total_disc_loss += loss_disc.item()
        
        ''' 2. Train Generator '''
        # Keep tracking active from the start of mapping generation
        w_list = apply_style_mixing(mapping_network, batch_size, latent_dim, device, mixing_prob)
        fake_images = synthesis_network(w_list)
        
        pred_fake = discriminator(fake_images)
        loss_gen = logistic_loss_generator(pred_fake)
        
        # Lazy Path Length Regularization
        pl_loss = torch.tensor(0.0, device=device)
        if step % pl_interval == 0:
            # We don't overwrite requires_grad manually here anymore; 
            # pl_regularizer handles it through the passed graph
            pl_loss, _ = pl_regularizer(w_list, fake_images)
            total_pl_loss += pl_loss.item()
            pl_count += 1
        
        # Total G Loss (scaled for lazy regularization execution)
        total_loss_gen = loss_gen + (lambda_pl * pl_loss * pl_interval)
        
        optim_mapping.zero_grad(set_to_none=True)
        optim_synthesis.zero_grad(set_to_none=True)
        total_loss_gen.backward()
        optim_mapping.step()
        optim_synthesis.step()

        mapping_ema.update(mapping_network)
        synthesis_ema.update(synthesis_network)
        
        total_gen_loss += loss_gen.item()
        step += 1
    
    num_batches = len(dataloader)
    return {
        'disc_loss': total_disc_loss / (num_batches * n_critics),
        'gen_loss': total_gen_loss / num_batches,
        'r1_loss': total_r1_loss / max(1, r1_count),
        'pl_loss': total_pl_loss / max(1, pl_count),
        'updated_step': step  
    }


def display_generated_images(mapping_network, synthesis_network, device, epoch, fixed_latents, num_images=16, figsize=(8, 8)):
    mapping_was_training = mapping_network.training
    synthesis_was_training = synthesis_network.training
    
    mapping_network.eval()
    synthesis_network.eval()
    
    with torch.no_grad():
        z = fixed_latents
        w = mapping_network(z) # Shape: [16, 256]
        
        # FIX FOR FIX 1: Manually broadcast the single tensor into a list of 12 layers
        w_list = [w for _ in range(12)] 
        
        fake_images = synthesis_network(w_list)
    
    fake_images = fake_images.cpu() * 0.5 + 0.5
    fake_images = torch.clamp(fake_images, 0, 1)
    
    grid_size = int(num_images ** 0.5)  
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        img = fake_images[idx].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    filename = f'./generated_images/epoch_{epoch:03d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')
    
    if mapping_was_training: mapping_network.train()
    if synthesis_was_training: synthesis_network.train()


def create_gif(image_folder="./generated_images", output_path="training.gif", duration=0.5):
    if not os.path.exists(image_folder):
        print(f"Error: {image_folder} does not exist")
        return
    
    images = []
    files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    
    if not files:
        print(f"No PNG files found in {image_folder}")
        return
    
    for file in files:
        img_path = os.path.join(image_folder, file)
        images.append(imageio.imread(img_path))
    
    imageio.mimsave(output_path, images, duration=duration)
    print(f"GIF saved to {output_path}")


def save_checkpoint(epoch, global_step, discriminator, mapping_network, synthesis_network, mapping_ema, synthesis_ema,
                    optim_discriminator, optim_mapping, optim_synthesis, filepath="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "discriminator": discriminator.state_dict(),
        "mapping_network": mapping_network.state_dict(),
        "synthesis_network": synthesis_network.state_dict(),
        "mapping_network": mapping_network.state_dict(),
        "mapping_network_ema": mapping_ema.ema_model.state_dict(),
        "synthesis_network_ema": synthesis_ema.ema_model.state_dict(),
        "synthesis_network": synthesis_network.state_dict(),
        "optim_discriminator": optim_discriminator.state_dict(),
        "optim_mapping": optim_mapping.state_dict(),
        "optim_synthesis": optim_synthesis.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(filepath, discriminator, mapping_network, synthesis_network,
                    mapping_ema, synthesis_ema,  # <-- Added EMA trackers
                    optim_discriminator, optim_mapping, optim_synthesis, device):
    if not os.path.exists(filepath):
        print("No checkpoint found. Starting from scratch.")
        return 0, 0

    checkpoint = torch.load(filepath, map_location=device)
    
    # Load raw network states
    discriminator.load_state_dict(checkpoint["discriminator"])
    mapping_network.load_state_dict(checkpoint["mapping_network"])
    synthesis_network.load_state_dict(checkpoint["synthesis_network"])
    
    # Load optimizer states
    optim_discriminator.load_state_dict(checkpoint["optim_discriminator"])
    optim_mapping.load_state_dict(checkpoint["optim_mapping"])
    optim_synthesis.load_state_dict(checkpoint["optim_synthesis"])

    # --- Load or Initialize EMA states ---
    if "mapping_network_ema" in checkpoint and "synthesis_network_ema" in checkpoint:
        mapping_ema.ema_model.load_state_dict(checkpoint["mapping_network_ema"])
        synthesis_ema.ema_model.load_state_dict(checkpoint["synthesis_network_ema"])
        print("Loaded EMA states from checkpoint.")
    else:
        # Fallback: If transitioning an old checkpoint to this new script,
        # initialize EMA using the freshly loaded raw weights.
        print("EMA states not found in checkpoint. Initializing EMA with current weights.")
        mapping_ema.update(mapping_network)
        synthesis_ema.update(synthesis_network)

    print(f"Resuming from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"], checkpoint.get("global_step", 0)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random_number_generator = torch.Generator().manual_seed(42)
    print('Torch version:', torch.__version__)
    print('Device used:', device)

    os.makedirs("generated_images", exist_ok=True) 

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMG_SIZE),
        torchvision.transforms.CenterCrop(IMG_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CelebA(root='./data', split='train', download=True, transform=transform)

    train_subset_size = int(len(trainset) * SUB_SET_SIZE)
    train_subset, _ = torch.utils.data.random_split(trainset, [train_subset_size, len(trainset) - train_subset_size], generator=random_number_generator)

    # displaying the number of examples in the training and validation sets
    print('Number of examples in the training set', len(trainset))
    print(f'Number of examples in the training subset({SUB_SET_SIZE*100}%)', len(train_subset))

    train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    discriminator = Discriminator().to(device, non_blocking=True)
    mapping_network = MappingNetwork().to(device, non_blocking=True)
    synthesis_network = SynthesisNetwork().to(device, non_blocking=True)

    mapping_ema = ExponentialMovingAverage(mapping_network, decay=0.999)
    synthesis_ema = ExponentialMovingAverage(synthesis_network, decay=0.999)

    # Mathematical adjustments for lazy regularization
    # Formula: LR_corrected = LR * (interval / (interval + 1))
    # Formula: Beta2_corrected = Beta2 ** interval
    lr_d_lazy = LR * (R1_INTERVAL / (R1_INTERVAL + 1))
    lr_g_lazy = LR * (PL_INTERVAL / (PL_INTERVAL + 1))

    beta2_d_lazy = 0.99 ** R1_INTERVAL
    beta2_g_lazy = 0.99 ** PL_INTERVAL

    optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr_d_lazy, betas=(0.0, beta2_d_lazy), eps=1e-8)
    optim_mapping = torch.optim.Adam(mapping_network.parameters(), lr=lr_g_lazy, betas=(0.0, beta2_g_lazy), eps=1e-8)
    optim_synthesis = torch.optim.Adam(synthesis_network.parameters(), lr=lr_g_lazy, betas=(0.0, beta2_g_lazy), eps=1e-8)


    start_epoch, global_step = load_checkpoint(checkpoint_path, discriminator, mapping_network, synthesis_network, 
                                               mapping_ema, synthesis_ema, optim_discriminator, optim_mapping, optim_synthesis, device)

    print('\nTraining')
    eval_generator = torch.Generator(device=device).manual_seed(42)
    fixed_latents = torch.randn(16, LATENT_DIM, device=device, generator=eval_generator)
    pl_regularizer = PathLengthRegularizer(decay=0.99).to(device)

    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()

        results = train_step(
            train_loader, discriminator, mapping_network, synthesis_network,
            optim_discriminator, optim_mapping, optim_synthesis, 
            LATENT_DIM, mapping_ema, synthesis_ema, device, N_CRITIC,
            pl_regularizer,
            lambda_r1=LAMBDA_R1,          
            lambda_pl=LAMBDA_PL,
            mixing_prob=MIXING_PROB,
            r1_interval=R1_INTERVAL,          
            pl_interval=PL_INTERVAL,
            global_step=global_step, 
        )

        total_time = time.time() - start_time
        global_step = results['updated_step']

        print(f"Epoch: {epoch+1:03d} | "
              f"D_loss: {results['disc_loss']:.4f} | "
              f"G_loss: {results['gen_loss']:.4f} | "
              f"R1: {results['r1_loss']:.4f} | "      
              f"PL (on step): {results['pl_loss']:.4f} | "
              f"Total Steps: {global_step} | "
              f"Time: {total_time:.2f}s")

        if (epoch + 1) % 2 == 0:
            save_checkpoint(epoch + 1, global_step, discriminator, mapping_network, synthesis_network,mapping_ema, synthesis_ema, optim_discriminator, optim_mapping, optim_synthesis, checkpoint_path)
            display_generated_images(mapping_ema.ema_model, synthesis_ema.ema_model, device, epoch + 1, fixed_latents=fixed_latents)
        
        # Explicitly empty CUDA cache after each epoch to avoid the autograd growth leak
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    create_gif(image_folder="./generated_images", output_path="training_progress.gif", duration=0.5)

if __name__ == '__main__':
    main()