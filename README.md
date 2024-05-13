# Setting up a Virtual Environment and Running a Streamlit Application

## Prerequisites
- Python 3.x installed on your machine
- pip package manager installed

## Step 1: Create a Virtual Environment
1. Open a terminal or command prompt.
2. Navigate to the project directory: `cd <your path>/project`.
3. Create a new virtual environment: `python3 -m venv myenv`.

## Step 2: Activate the Virtual Environment
1. Activate the virtual environment:
    - On macOS/Linux: `source myenv/bin/activate`.
    - On Windows: `myenv\Scripts\activate.bat`.

## Step 3: Install Dependencies
1. Install the required packages using pip:
    ```shell
    pip install -r requirements.txt
    ```

## Step 4: Run the Streamlit Application
1. Start the Streamlit application:
    ```shell
    streamlit run app.py
    ```
    Replace `app.py` with the name of your Streamlit application file.

2. Open a web browser and navigate to `http://localhost:8501` to view your Streamlit application.

## Step 5: Deactivate the Virtual Environment
1. When you're done working with the virtual environment, you can deactivate it:
    ```shell
    deactivate
    ```

That's it! You have successfully set up a virtual environment and run a Streamlit application.