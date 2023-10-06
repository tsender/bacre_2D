from logging.config import valid_ident
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
import tensorflow as tf
import cv2
import math
import os
import glob
import multiprocessing
from typing import Tuple, List, Callable
    

def create_training_masks(seg_files: List[str], image_size: Tuple[int], dir: str, semantic_color_map: Tuple[Tuple[int]]):
        """Creates and saves training masks
        
        Args:
            - seg_files: List of file names of the semantic segmentation images
            - image_size: Image size
            - dir: Directory to save images to
            - semantic_color_map: Semantic color map, tuple of (R,G,B) tuples
        """
        print("Creating training masks...")
        count = 0
        for seg_file_name in seg_files:
            suffix = seg_file_name.split('_')[-1]
            # number = suffix.split('.')[0]
            img = cv2.imread(seg_file_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            assert image_size == img.shape
            mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint16)

            # Loop over pixels and create mask
            for i in range(image_size[0]):
                for j in range(image_size[1]):
                    for k, cmap in enumerate(semantic_color_map):
                        if np.array_equal(img[i,j,:], cmap):
                            mask[i,j] = k
                            # print(f"adding label {k}")

            # print(f"Max label = {np.max(mask)}")
            mask_file_name = os.path.join(dir, 'mask_' + suffix)
            cv2.imwrite(mask_file_name, mask)
            count += 1
            print(f"created {'mask_' + suffix}")

class SemanticSegmentation:
    def __init__(self,
                mode: str,                                          # Mode of operation: "training" or "prediction"
                data_dir: str,                                      # Data directory for accessing image
                model_version: str,                                 # Indicates which model to load/stor network model
                image_size: Tuple[int],                             # Image size (H,W,C)
                semantic_color_map: Tuple[Tuple[int]],              # Semantic color map, tuple of (R,G,B) tuples, where the class label is determined from the index
                create_network_func: Callable[[Tuple[int], int], tf.keras.Model],   # Callable to create model network
                seg_folder_name: str = 'seg',                      # Folder name for segmentation images inside the training data folder
                learning_rate: float = 0.0001,                      # Learning rate to use for training
                batch_size: int = 16,                               # Batch size to use for training
                num_epochs: int = 5,                                # Number of epochs used for training
                validation_fraction: float = 0.1,                   # Fraction of all images to be used solely for validation
                ):                                
        possible_modes = ['training', 'prediction', 'validation']
        if mode not in possible_modes:
            raise Exception(f'Input mode {mode} is not in the list of possible modes {possible_modes}')
        self.mode = mode

        self.image_size = image_size
        self.create_model_func = create_network_func
        self.semantic_color_map = semantic_color_map
        assert 0 < len(semantic_color_map) <= 2**16 # For now, we are limiting to 2^16 labels which should be sufficient
        self.num_labels = len(semantic_color_map)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_fraction = validation_fraction

        # Create directories
        self.seg_folder_name = seg_folder_name
        self.data_dir = data_dir
        self.main_dir = os.path.join(data_dir, 'models', model_version)
        self.dirs = {}
        self.dirs['main'] = self.main_dir
        self.dirs['checkpoint'] = os.path.join(self.main_dir, 'checkpoints')
        self.dirs['training_data_color'] = os.path.join(self.data_dir, 'color')
        self.dirs['training_data_seg'] = os.path.join(self.data_dir, seg_folder_name)
        self.dirs['training_data_mask'] = os.path.join(self.data_dir, 'mask')
        self.dirs['seg_prediction'] = os.path.join(self.main_dir, 'seg_prediction')
        for _,dir in self.dirs.items():
            if not os.path.isdir(dir):
                os.makedirs(dir)

        self.valid_idx_filename = os.path.join(self.main_dir, 'validation_idx.csv')
        
        if self.mode == 'training':
            color_files = glob.glob(os.path.join(self.dirs['training_data_color'], '*'))
            seg_files = glob.glob(os.path.join(self.dirs['training_data_seg'], '*'))
            mask_files = glob.glob(os.path.join(self.dirs['training_data_mask'], '*'))

            # Check for mask directory and create masks if needed
            assert len(color_files) == len(seg_files)
            if len(color_files) > 0 and len(color_files) != len(mask_files):
                # Use multiprocessing to reduce runtime
                num_workers = 10
                
                # Split seg_file list into num_workers chunks
                chunk_size = math.ceil(len(seg_files) / num_workers)
                seg_file_chunks = []
                for i in range(0, len(seg_files), chunk_size):
                    seg_file_chunks.append(seg_files[i:i+chunk_size])

                # Create processes
                process_list = []
                for i in range(num_workers):
                    p = multiprocessing.Process(target = create_training_masks, args = (seg_file_chunks[i], image_size, self.dirs['training_data_mask'], semantic_color_map))
                    p.start()
                    process_list.append(p)
                for process in process_list:
                    process.join()

            # Create list of training and validation files
            print("Creating training and validation groups...")
            self.num_valid_samples = round(len(color_files) * self.validation_fraction)
            self.num_training_samples = len(color_files) - self.num_valid_samples
            val_idx = np.random.choice(len(color_files), size=self.num_valid_samples, replace=False)

            # Save csv file with indeces for validation files
            np.savetxt(self.valid_idx_filename, val_idx.reshape((self.num_valid_samples,1)), delimiter=',')

            # Sort files into training/validation groups
            color_files = glob.glob(os.path.join(self.dirs['training_data_color'], '*'))
            mask_files = glob.glob(os.path.join(self.dirs['training_data_mask'], '*')) # Reload mask files (in case they did not exist previously)
            train_color_files = []
            valid_color_files = []
            valid_mask_files = []
            
            for i in range(len(color_files)):
                if i in val_idx:
                    valid_color_files.append(color_files[i])
                    # valid_mask_files.append(mask_files[i])
                else:
                    train_color_files.append(color_files[i])

            # Load training color and mask files into tf.Dataset
            print("Creating tf.Dataset...")
            self.dataset = tf.data.Dataset.list_files(train_color_files).map(self._parse_image, num_parallel_calls=8).shuffle(len(train_color_files)).repeat().batch(self.batch_size)
            self.valid_dataset = tf.data.Dataset.list_files(valid_color_files).map(self._parse_image_no_flip, num_parallel_calls=8).batch(1)
            self.dataset = iter(self.dataset)
            self.valid_dataset = iter(self.valid_dataset)

            self.num_batches_per_epoch = math.ceil(self.num_training_samples / self.batch_size)

        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
        self.model = create_network_func(image_size, self.num_labels)
        self.model.summary()

        self.checkpoint = tf.train.Checkpoint(model=self.model, opt=self.opt)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.dirs['checkpoint'], max_to_keep=5)

        if self.mode == 'prediction':
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print("Restored to latest checkpoint")
            self.dummy_cudnn_init()

        print("Semantic Segmentation initialized")

    @tf.function
    def _parse_image(self, img_path: str):
        """Called by tf.Dataset to read file path to the color image and then return a normalized image and its mask/label. Randomly flips 50% of all images.

        Args:
            - img_path: File path to the image

        Returns:
            The normalized image and its mask
        """
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0

        mask_path = tf.strings.regex_replace(img_path, "color", "mask")
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1, dtype=tf.uint16)

        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        return image, mask

    @tf.function
    def _parse_image_no_flip(self, img_path: str):
        """Called by tf.Dataset to read file path to the color image and then return a normalized image and its mask/label

        Args:
            - img_path: File path to the image

        Returns:
            The normalized image and its mask
        """
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0

        mask_path = tf.strings.regex_replace(img_path, "color", "mask")
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1, dtype=tf.uint16)

        return image, mask

    @tf.function
    def _train_step(self, images: tf.float32, masks: tf.uint16):
        with tf.GradientTape() as tape:
            predicted_probs = self.model(images, training=True)
            loss = self.loss(masks, predicted_probs)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self):
        if not self.mode == 'training':
            print("Not in training mode.")
            return

        print("Training...")
        train_losses = []
        for epoch in range(self.num_epochs):
            for b in range(self.num_batches_per_epoch):
                batch = self.dataset.next()
                image_batch, mask_batch = batch
                loss = self._train_step(image_batch, mask_batch)
                train_losses.append(loss)

                print(f"Iter {epoch * self.num_batches_per_epoch + b + 1} / {self.num_epochs*self.num_batches_per_epoch}: Loss = {loss:.6f}")
            checkpoint_path = self.checkpoint_manager.save()
            print(f"Saving checkpoint to path: {checkpoint_path}")

        model_path = os.path.join(self.main_dir, 'trained_model.h5')
        self.model.save(model_path)

        # Compute validation loss and accuracy
        valid_losses = []
        num_pixel_labels_correct = []
        for valid_data in self.valid_dataset:
            image, mask = valid_data # Tensors in the form of [B,H,W,C]
            predicted_mask_probs = self.model(image)
            valid_losses.append(self.loss(mask, predicted_mask_probs).numpy())

            # Accuracy per pixel
            predicted_mask = tf.math.argmax(predicted_mask_probs[0], axis=-1).numpy() # Of size HxW
            mask_np = mask[0].numpy().reshape((self.image_size[0], self.image_size[1]))
            num_correct = np.sum(np.equal(predicted_mask, mask_np))
            num_pixel_labels_correct.append(num_correct)

        valid_loss = np.mean(valid_losses) # Avg pixel loss
        valid_acc = np.sum(num_pixel_labels_correct) / (self.image_size[0] * self.image_size[1] * len(num_pixel_labels_correct))

        # Write params to file
        print("Writing params to file...")
        with open(os.path.join(self.main_dir, "training_info.txt"), "w") as f:
            f.write(f"Semantic Color Map: {self.semantic_color_map}" + "\n")
            f.write(f"Dataset Size: {self.num_training_samples + self.num_valid_samples} Total, {self.num_training_samples} Train / {self.num_valid_samples} Valid" + "\n")
            f.write(f"Validation Fraction: {self.validation_fraction}" + "\n")
            f.write(f"Learning Rate: {self.learning_rate}" + "\n")
            f.write(f"Batch Size: {self.batch_size}" + "\n")
            f.write(f"Training Epochs: {self.num_epochs}" + "\n")
            f.write(f"Num Training Batches: {self.num_batches_per_epoch * self.num_epochs}" + "\n")
            f.write(f"Validation Loss (avg pixel loss): {valid_loss:.6f}" + "\n")
            f.write(f"Validation Accuracy (avg pixel label acc): {valid_acc:.6f}" + "\n")

        # Save training losses to csv
        print("Saving training losses to csv...")
        loss_file = os.path.join(self.main_dir, "training_losses.csv")
        train_losses = np.array(train_losses)
        np.savetxt(loss_file, train_losses.reshape((len(train_losses), 1)), delimiter=',')

        fig, ax = plt.subplots(num=1)
        ax.plot(list(range(1, 1+len(train_losses))), train_losses)
        ax.set_xlabel("Batch Number")
        ax.set_ylabel("Training Loss")
        plt.savefig(os.path.join(self.main_dir, "training_losses.pdf"))
        plt.savefig(os.path.join(self.main_dir, "training_losses.png"))
        plt.cla()
        plt.clf()

        print("Done.")
        
    def generate_segmentation_image(self, mask: np.uint16):
        """Create RGB semantic image from the given mask
        
        Args:
            - mask: A mask indicating the label for each pixel

        Returns:
            The corresponding semantic segmentation image
        """
        image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                image[i,j,:] = self.semantic_color_map[mask[i,j]]
                # print(f"color map {mask[i,j]}")

        return image

    @tf.function
    def predict_mask_probs(self, image: tf.float32):
        return self.model(image)

    def generate_predictions(self, image: np.uint8):
        """Generate predicted mask probabilities and predicted mask
        
        Args:
            - image: A single RGB image to use when generating predictions (we require one input image to minimize GPU memory)

        Returns
            Predicted mask probabilities (3D array) and the argmax (2D array)
        """
        image = image[np.newaxis, :]
        image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.
        probs = self.predict_mask_probs(image)[0].numpy()
        mask = tf.math.argmax(probs, axis=-1).numpy()
        return probs, mask

    def predict_from_files(self, num_to_predict: int):
        """Predict labels"""
        color_files = glob.glob(os.path.join(self.dirs['training_data_color'], '*'))
        idx = np.random.choice(len(color_files), num_to_predict, replace=False).tolist()

        for i in idx:
            color_file = color_files[i]
            suffix = color_file.split('_')[-1]
            image = cv2.imread(color_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            probs, mask = self.generate_predictions(image)
            seg_image = self.generate_segmentation_image((mask))
            seg_image = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)
            seg_file = os.path.join(self.dirs['seg_prediction'], 'seg_prediction_' + suffix)
            cv2.imwrite(seg_file, seg_image)
            print(f"Saved semantic prediction: {'seg_prediction_' + suffix}")

    def dummy_cudnn_init(self):
        """Pass a few sample images through the network to initialize cudnn (so it doesn't take time later)"""
        color_files = glob.glob(os.path.join(self.dirs['training_data_color'], '*'))
        idx = np.random.choice(len(color_files), 2, replace=False).tolist()

        for i in idx:
            color_file = color_files[i]
            suffix = color_file.split('_')[-1]
            image = cv2.imread(color_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            probs, mask = self.generate_predictions(image)

if __name__ == "__main__":
    import semseg_networks

    # Set memory growth to True
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    semantic_color_map = ((255, 255, 255), # 0 Traversable
                            (0, 0, 0),      # 1 Non-traversable
                            (0, 0, 255))    # 2 Sky

    mode = 'training'
    semseg_dict = {
        'mode': mode,       # Possible modes: training, prediction, validation
        'data_dir': os.path.join(os.getcwd(), "Datasets_256x256", "NoClouds_Trees_Bushes_SunIncl0-15-30-45-60-75-90"),
        'model_version': 'small_v2_test',
        'image_size': (256, 256, 3),
        'semantic_color_map': semantic_color_map,
        'create_network_func': semseg_networks.create_unet_network_small_v2,
        'seg_folder_name': 'trav',
        'learning_rate': 0.0001,
        'batch_size': 16,
        'num_epochs': 4,
        'validation_fraction': 0.1,
    }
    SemSeg = SemanticSegmentation(**semseg_dict)
    if mode == 'training':
        SemSeg.train()
    elif mode == 'prediction':
        SemSeg.predict_from_files(10)