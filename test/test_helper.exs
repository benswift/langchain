# Load the ENV key for running live OpenAI tests.
Application.put_env(:langchain, :openai_key, System.fetch_env!("OPENAI_KEY"))
Application.put_env(:langchain, :replicate_key, System.fetch_env!("REPLICATE_KEY"))

ExUnit.configure(capture_log: true, exclude: [live_call: true])
ExUnit.start()
