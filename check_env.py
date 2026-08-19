import sys
print(sys.version)
try:
    import langchain
    print("langchain version:", langchain.__version__)
except Exception as e:
    print("langchain not found", e)
try:
    from langchain_community.utilities import SQLDatabase
    print("SQLDatabase found in langchain_community")
except Exception as e:
    print("SQLDatabase not found in langchain_community", e)
