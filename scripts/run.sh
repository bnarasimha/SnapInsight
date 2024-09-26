#!/bin/bash

# Detect the OS
os_type=$(uname)

# Function to install doctl
install_doctl() {
    if [[ "$os_type" == "Darwin" ]]; then
        echo "Installing doctl via Homebrew..."
        brew install doctl
    elif [[ "$os_type" == "Linux" ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            if [[ "$ID" == "ubuntu" ]]; then
                echo "Installing doctl via Snap..."
                sudo snap install doctl
            else
                echo "Please install doctl manually for your Linux distribution."
                exit 1
            fi
        else
            echo "Unable to determine Linux distribution. Please install doctl manually."
            exit 1
        fi
    else
        echo "Unsupported OS for automatic doctl installation. Please install doctl manually."
        exit 1
    fi
}

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    echo "doctl is not installed. Installing now..."
    install_doctl
else
    echo "doctl is already installed."
fi

# Prompt for required information
read -p "Enter droplet name: " droplet_name
read -p "Enter SSH key fingerprint: " ssh_fingerprint
read -p "Enter Hugging Face token: " huggingface_token


# Check if the droplet already exists
existing_droplet=$(doctl compute droplet list --format ID,Name --no-header | grep "$droplet_name")

if [ -n "$existing_droplet" ]; then
    echo "Droplet '$droplet_name' already exists. Using the existing droplet."
else
    # Create GPU Droplet
    echo "Creating GPU Droplet..."
    doctl compute droplet create "$droplet_name" \
        --region tor1 \
        --image gpu-h100x1-base \
        --size gpu-h100x1-80gb \
        --ssh-keys "$ssh_fingerprint"
fi

# Wait for the droplet to be active
echo "Waiting for the GPU Droplet to be active..."
while true; do
    status=$(doctl compute droplet get "$droplet_name" --format Status --no-header)
    if [ "$status" = "active" ]; then
        echo "GPU Droplet is active."
        break
    fi
    echo "Current status: $status. Waiting..."
    sleep 10
done

# Get IP Address of GPU Droplet
echo "Getting IP address of the GPU Droplet..."
gpu_droplet_ip=$(doctl compute droplet get "$droplet_name" --format PublicIPv4 --no-header)

echo "GPU Droplet IP: $gpu_droplet_ip"

# Wait for SSH to become available
echo "Waiting for SSH service to start..."
while ! nc -z "$gpu_droplet_ip" 22; do
  echo "SSH not available yet. Waiting..."
  sleep 5
done
echo "SSH is now available."

# Add a small delay to ensure SSH is fully ready
sleep 10

# SSH into the GPU Droplet and set up the environment
echo "Setting up the GPU Droplet..."
ssh -o StrictHostKeyChecking=no root@"$gpu_droplet_ip" << EOF
    
    # Clone the SnapInsight repository
    git clone https://github.com/bnarasimha/SnapInsight.git
    cd SnapInsight

    apt update
    apt install -y python3.10-venv

    python3 -m venv venv
    source venv/bin/activate

    pip install -r requirements.txt

    # Set Hugging Face token as an environment variable
    export HUGGING_FACE_HUB_TOKEN="$huggingface_token"

    # Log in to Hugging Face (this should use the token from the environment variable)
    python -c "from huggingface_hub.hf_api import HfApi; HfApi().whoami()"

    ufw allow 22
    ufw allow 7860
    echo "y" | ufw enable

    nohup python3 app.py > app.log 2>&1 &

    echo "Application is running. You can access it at http://$gpu_droplet_ip:7860"

    # Clean up Git credentials
    rm ~/.git-credentials
    git config --global --unset credential.helper
EOF

echo "Setup complete. You can SSH into the droplet using: ssh root@$gpu_droplet_ip"

