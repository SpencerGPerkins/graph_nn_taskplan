import subprocess

def run_pyperplan(domain_file, problem_file):
    command = ["python3", "-m", "pyperplan", domain_file, problem_file]
    result = subprocess.run(command, capture_output=True, text=True)

if __name__ == "__main__":
    domain_path = "domain.pddl"  
    problem_path = "problem.pddl"  
    run_pyperplan(domain_path, problem_path)







