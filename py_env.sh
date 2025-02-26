#!/bin/bash
# Script to set up a Python virtual environment for knowledge distillation

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up Python environment for knowledge distillation...${NC}"

# Create a directory for the project
PROJECT_DIR="knowledge_distillation_project"
echo -e "${YELLOW}Creating project directory: ${PROJECT_DIR}${NC}"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python 3 not found. Please install Python 3.8 or newer.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}Using Python version: ${PYTHON_VERSION}${NC}"

# Check if the version is 3.8 or greater
if [[ $(echo $PYTHON_VERSION | cut -d'.' -f1,2 | sed 's/\.//') -lt 38 ]]; then
    echo -e "${YELLOW}Warning: Recommended Python version is 3.8 or newer${NC}"
fi

# Create a virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv kd_env

# Activate the virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source kd_env/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install required packages
echo -e "${YELLOW}Installing required packages...${NC}"
pip install torch torchvision
pip install transformers datasets
pip install nltk tqdm requests numpy
pip install matplotlib pandas seaborn
pip install scikit-learn

# Clone the knowledge distillation code
echo -e "${YELLOW}Setting up knowledge distillation code...${NC}"

# Create main knowledge distillation script
cat > knowledge_distillation.py << 'EOL'
# Knowledge distillation implementation will be placed here
# Copy and paste the full code from the artifact provided earlier
EOL

# Create data generation script
cat > create_training_data.py << 'EOL'
# Training data generation code will be placed here
# Copy and paste the full code from the artifact provided earlier
EOL

# Create a simple README
cat > README.md << 'EOL'
# Knowledge Distillation Project

This project implements knowledge distillation from a pre-trained LLM (teacher) to a single-layer transformer decoder (student).

## Setup

1. Activate the virtual environment:
   ```
   source kd_env/bin/activate
   ```

2. Generate training data:
   ```
   python create_training_data.py --output_dir data --dataset_type diverse --size 100
   ```

3. Run knowledge distillation:
   ```
   python knowledge_distillation.py --teacher gpt2 --data data/combined_data_train.txt
   ```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Additional dependencies are installed in the virtual environment
EOL

# Create a helper script to activate the environment
cat > activate_env.sh << 'EOL'
#!/bin/bash
source kd_env/bin/activate
echo "Knowledge distillation environment activated. Run 'deactivate' to exit."
EOL

chmod +x activate_env.sh

# Create a helper script for training
cat > run_training.sh << 'EOL'
#!/bin/bash
source kd_env/bin/activate

# Check if data directory exists, if not, create it
if [ ! -d "data" ]; then
    echo "Creating data directory and generating training data..."
    python create_training_data.py --output_dir data --dataset_type diverse --size 50
fi

# Run the knowledge distillation
python knowledge_distillation.py --teacher gpt2 --data data/combined_data_train.txt --epochs 3 --batch_size 16

echo "Training complete. Check the saved model at student_model.pt"
EOL

chmod +x run_training.sh

echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${GREEN}To activate the environment, run: source kd_env/bin/activate${NC}"
echo -e "${GREEN}Or use the helper script: ./activate_env.sh${NC}"
echo -e "${GREEN}To start training, run: ./run_training.sh${NC}"
echo -e "${BLUE}Project directory: $(pwd)/${PROJECT_DIR}${NC}"