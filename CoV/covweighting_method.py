from CoV import covweighting_loss
from CoV.scheduler import LearningRateScheduler


class CoVWeightingMethod:
    def __init__(self, device, mode, epochs, adjust_lr, lr_mode, learning_rate, num_losses, b_mean_decay):
        self.device = device
        self.mode = mode

        self.losses = {}
        self.epochs = epochs

        self.learning_rate_scheduler = None
        self.criterion = None

        # Set the optimizer and scheduler, but wait for method-specific parameters.
        if self.mode == 'train':
            self.learning_rate_scheduler = LearningRateScheduler(adjust_lr, lr_mode, learning_rate,
                                                                 self.epochs)
            self.criterion = covweighting_loss.CoVWeightingLoss(device=device, b_train=True, num_losses=num_losses, b_mean_decay=b_mean_decay)
            self.criterion.to(self.device)

            # Record the mean weights for an epoch.
            self.mean_weights = [0.0 for _ in range(self.criterion.alphas.shape[0])]

    def set_num_losses(self, num_losses):
        self.criterion.set_num_losses(num_losses)

    def run_epoch(self, current_epoch, loss, optimizer):
        # First, adjust learning rate.
        self.update_learning_rate(current_epoch, optimizer)
        # Then optimize.
        train_loss = self.optimize_parameters(loss, optimizer)

        # Record the running loss.
        if current_epoch not in self.losses:
            self.losses[current_epoch] = {}
        self.losses[current_epoch]['train'] = train_loss

        return train_loss

    def optimize_parameters(self, loss, optimizer):
        loss = self.criterion.forward(loss)
        print("    LOSS:    ", loss)
        loss.backward()
        print("    LOSS BACKWARDED:    ", loss)
        optimizer.step()

        # Finally, add the scales to the mean scales to get an idea of the mean weights after training.
        for i, weight in enumerate(self.criterion.alphas):
            self.mean_weights[i] += weight.item()
        return loss

    def update_learning_rate(self, current_epoch, optimizer):
        self.learning_rate_scheduler(optimizer, current_epoch)
