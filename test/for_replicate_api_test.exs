defmodule LangChain.ForReplicateApiTest do
  use ExUnit.Case
  doctest LangChain.ForReplicateApi
  alias LangChain.ForReplicateApi
  alias LangChain.Message

  describe "for_api/1" do
    test "turns a system message into expected string format" do
      msg = Message.new_system!("You are a helpful assistant.")

      api_msg = ForReplicateApi.for_api(msg)

      assert api_msg == "You are a helpful assistant."
    end

    test "turns a user message into expected string format" do
      msg = Message.new_user!("Testing 1, 2, 3")

      api_msg = ForReplicateApi.for_api(msg)

      assert api_msg == "Testing 1, 2, 3"
    end

    test "turns an assistant message into expected string format" do
      msg = Message.new_assistant!("Ok, I can do that for you.")

      api_msg = ForReplicateApi.for_api(msg)

      assert api_msg ==  "[INST] Ok, I can do that for you. [/INST]"
    end
  end
end
