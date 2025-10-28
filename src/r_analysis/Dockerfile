# Use the official R base image from Docker Hub
FROM r-base:4.3.3 

# Set the working directory in the container
WORKDIR /app

# Copy the package installation script into the container
COPY src/r_analysis/install_packages.R .

# Run the installation script
RUN Rscript install_packages.R

# Copy the entire project context into the container
COPY . .

# Run the R script when the container launches
CMD ["Rscript", "src/r_analysis/main.R"]