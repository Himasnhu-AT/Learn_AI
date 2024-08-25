# Setting up the Python Environment

## Introduction

This tutorial will guide you through setting up your Python environment for machine learning. We'll be using Anaconda, a popular distribution that bundles Python with essential libraries and tools. By the end of this tutorial, you'll have a functional workspace for your machine learning journey.

## Installing Anaconda

1. **Download Anaconda:** Visit the Anaconda website ([https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)) and download the installer for your operating system (Windows, macOS, or Linux).
2. **Run the installer:** Follow the on-screen instructions to install Anaconda. Make sure to add Anaconda to your system's PATH environment variable during the installation process. This allows you to run Anaconda commands from any terminal or command prompt.
3. **Verify the installation:** Open your terminal or command prompt and type `conda --version`. If the installation was successful, you should see the version of conda displayed.

## Using Anaconda Prompt

Anaconda comes with a dedicated command prompt called "Anaconda Prompt". This prompt is where you'll execute most commands related to your Python environment.

1. **Open Anaconda Prompt:** Search for "Anaconda Prompt" in your Start menu (Windows) or Applications folder (macOS).
2. **Update Anaconda:** It's always a good practice to keep your Anaconda environment up-to-date. In the Anaconda Prompt, run the command:
   ```bash
   conda update conda
   ```
3. **List installed packages:** To see the packages currently installed in your environment, use the following command:
   ```bash
   conda list
   ```

## Jupyter Notebook - Your Interactive Coding Environment

Jupyter Notebook is a powerful tool for interactive coding, especially for data science and machine learning. It allows you to write code, run it, and see the results immediately.

1. **Launch Jupyter Notebook:** In the Anaconda Prompt, type:
   ```bash
   jupyter notebook
   ```
   This will open a new browser window with the Jupyter Notebook dashboard.
2. **Create a new notebook:** From the dashboard, click on "New" and select "Python 3". This will create a new Jupyter Notebook file with a `.ipynb` extension.
3. **Run your first code:** In the notebook's code cell, type:
   Sample Python Code: 

```{language}
   print("Hello, World!")
   ```
   Run the code by pressing `Shift+Enter`. You should see the output "Hello, World!" displayed below the code cell.

## Basic Jupyter Notebook Commands

* **Creating a new cell:** Click on the "+" button in the toolbar to add a new code cell.
* **Running a cell:** Press `Shift+Enter` to run the code in the current cell.
* **Deleting a cell:** Select the cell you want to delete and press `dd` (delete twice).
* **Saving a notebook:** Click on the "File" menu and select "Save Notebook".

## Installing Additional Packages

If you need a package that isn't included in Anaconda's default installation, you can install it using `conda` or `pip`.

* **Using conda:**
   ```bash
   conda install <package_name>
   ```
* **Using pip:**
   ```bash
   pip install <package_name>
   ```

## Conclusion

You have now successfully set up your Python environment using Anaconda and Jupyter Notebook. You're ready to start exploring the world of machine learning!

## Assignment

1. **Create a new Jupyter Notebook file:**  Using the skills you learned, create a new Jupyter Notebook file and save it.
2. **Explore Jupyter Notebook:**  Experiment with creating, deleting, and running code cells in your new notebook. 
3. **Install a new package:**  Use `conda` or `pip` to install a package you're interested in, such as `numpy`. 
4. **Run a simple code:** In your notebook, write code to perform a basic calculation and print the result. For example, calculate and print the sum of two numbers.

By completing these assignments, you'll solidify your understanding of the environment setup and start getting familiar with the powerful tools of Jupyter Notebook. 
