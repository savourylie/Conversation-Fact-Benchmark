#!/bin/bash

# Setup script for Conversation Fact Benchmark datasets
# Run this script when setting up the project on a new machine

echo "Setting up datasets from HuggingFace..."

# Create datasets directory if it doesn't exist
mkdir -p datasets

# Instructions for manual setup (replace with your actual HuggingFace repo URLs)
echo ""
echo "Please manually clone your HuggingFace dataset repositories:"
echo ""
echo "cd datasets"
echo "git clone https://huggingface.co/datasets/YOUR_USERNAME/dream"
echo "git clone https://huggingface.co/datasets/YOUR_USERNAME/truthful_qa"
echo ""
echo "Or if using HuggingFace CLI:"
echo "huggingface-cli repo create YOUR_USERNAME/dream --type dataset"
echo "huggingface-cli repo create YOUR_USERNAME/truthful_qa --type dataset"
echo ""
echo "Make sure you have Git LFS installed: git lfs install"
echo ""
echo "After cloning, your datasets will be ready to use!" 