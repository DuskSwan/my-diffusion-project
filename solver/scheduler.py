import matplotlib.pyplot as plt

from diffusers import DDPMScheduler

def build_scheduler(max_steps=1000):
    # Initialize the DDPMScheduler with the desired parameters
    noise_scheduler = DDPMScheduler(num_train_timesteps=max_steps)
    return noise_scheduler

def show_scheduler(noise_scheduler):
    plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
    plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
    plt.legend(fontsize="x-large")
    plt.xlabel("Timesteps", fontsize="x-large")
    plt.ylabel("Value", fontsize="x-large")
    plt.title("DDPM Scheduler", fontsize="x-large")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    scheduler = build_scheduler(max_steps=1000)
    show_scheduler(scheduler)