import matplotlib.pyplot as plt

def generate_loss_curves(train_loss_dict, val_loss_dict, dir_name, max, file_name='loss_curves'):    

    # Create a figure with 1 row and 2 columns of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot each type of training loss on the left subplot
    for loss_type in train_loss_dict.keys():
        ax1.plot(train_loss_dict[loss_type], label=f'Training {loss_type}')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_ylim([0, max])

    # Plot each type of validation loss on the right subplot
    for loss_type in val_loss_dict.keys():
        ax2.plot(val_loss_dict[loss_type], label=f'Validation {loss_type}')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.set_ylim([0, max])

    # Show the plot
    plt.tight_layout()

    plt.savefig(f'{dir_name}/{file_name}')