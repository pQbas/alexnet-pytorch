FROM pytorch/pytorch

# Install pytest
RUN pip install pytest

# Set the working directory (optional)
WORKDIR /test

# Use the same command you use to start the container
CMD ["bash"]
