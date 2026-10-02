import subprocess
import tempfile
import os

class SepAdderFileReader:
    def __init__(self, path_to_jar: str):
        """
        Initializes the SepAdder with the path to the executable JAR file.
        """
        if not os.path.exists(path_to_jar):
            raise FileNotFoundError(f"JAR file not found at: {path_to_jar}")
        self.path_to_jar = path_to_jar

    def run_separator_jar(self, code: str) -> str:
        """
        Runs the Java JAR process by writing the input code to a temporary
        file and reading the result from another temporary file.
        This approach avoids stdout buffer limits for large outputs.
        """
        input_temp_file = None
        output_temp_file = None
        
        try:
            # Create a temporary file for the input code
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".java", encoding='utf-8') as temp_in:
                input_temp_file = temp_in.name
                temp_in.write(code)
                temp_in.flush() # Ensure data is written to disk

            # Create a temporary file to receive the output
            with tempfile.NamedTemporaryFile(mode='r', delete=False, suffix=".txt", encoding='utf-8') as temp_out:
                output_temp_file = temp_out.name

            # Command to execute the JAR, passing both input and output file paths
            command = [
                'java', '-jar', self.path_to_jar,
                input_temp_file,
                output_temp_file
            ]
            
            # Run the subprocess
            # We still capture output to see status messages or errors from the JAR
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True  # This will raise CalledProcessError if the JAR exits with a non-zero status
            )

            # Print status messages from the JAR for debugging
            print("Output from JAR (stdout):", result.stdout)
            if result.stderr:
                print("Errors from JAR (stderr):", result.stderr)

            # Read the processed code from the output temporary file
            with open(output_temp_file, 'r', encoding='utf-8') as f:
                processed_code = f.read()
            
            return processed_code

        except subprocess.CalledProcessError as e:
            print("Error executing the JAR file.")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return Code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise  # Re-raise the exception after printing details
        
        finally:
            # Clean up the temporary files
            if input_temp_file and os.path.exists(input_temp_file):
                os.remove(input_temp_file)
            if output_temp_file and os.path.exists(output_temp_file):
                os.remove(output_temp_file)

    def build_input(self, code: str, get_tokens: bool = False):
        """
        Processes the given code using the Java tagger.

        Args:
            code (str): The Java code to process.
            get_tokens (bool): If True, returns a list of space-separated tokens.
                               If False, returns the full processed string.

        Returns:
            Union[str, List[str]]: The processed code as a string or a list of tokens.
        """
        result_string = self.run_separator_jar(code)
        
        if get_tokens:
            return result_string.split()
        else:
            return result_string