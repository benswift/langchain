defmodule LangChain.ChatModels.ChatReplicateTest do
  use LangChain.BaseCase
  import LangChain.Fixtures

  doctest LangChain.ChatModels.ChatReplicate
  alias LangChain.ChatModels.ChatReplicate

  @live_call_model "meta/llama-2-7b-chat"

  describe "new/1" do
    test "works with minimal attr" do
      version = "aabbccdd"

      assert {:ok, %ChatReplicate{} = replicate} =
               ChatReplicate.new(%{"model" => @live_call_model, "version" => version})

      assert replicate.model == @live_call_model
      assert replicate.version == version
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatReplicate.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end
  end

  describe "for_api/3" do
    test "generates a map for an API call" do
      version = "aabbccdd"

      {:ok, replicate} =
        ChatReplicate.new(%{
          "model" => @live_call_model,
          "version" => version,
          "temperature" => 1
        })

      json = ChatReplicate.for_api(replicate, [], [])
      assert json.version == version
      assert json.input.temperature == 1
      assert json.input.prompt == ""
    end
  end

  describe "Req testing" do
    @tag :live_call
    test "" do
      version = "ac944f2e49c55c7e965fc3d93ad9a7d9d947866d6793fb849dd6b4747d0c061c"

      {:ok, resp} =
        Req.new(base_url: "https://api.replicate.com/v1")
        |> ChatReplicate.put_auth_header()
        |> Req.post(
          url: "/predictions",
          json: %{version: version, input: %{prompt: "this is a Req test..."}}
        )

      assert %Req.Response{body: %{"id" => id}} = resp
      assert String.match?(id, ~r/^[[:alnum:]]+$/)
    end
  end

  describe "call/2" do
    @tag :live_call
    test "get latest version of model from Replicate API" do
      {:ok, version} = ChatReplicate.latest_version(@live_call_model)

      # not 100% sure what contract there is on model ids, but this check'll do
      # for now
      assert String.match?(version, ~r/^[[:xdigit:]]+$/)
    end

    @tag :live_call
    test "basic content example" do
      # set_fake_llm_response({:ok, Message.new_assistant("\n\nRainbow Sox Co.")})

      # https://js.langchain.com/docs/modules/models/chat/
      {:ok, version} = ChatReplicate.latest_version(@live_call_model)
      {:ok, chat} = ChatReplicate.new(%{model: @live_call_model, version: version})

      {:ok, %Message{role: :assistant, content: response}} =
        ChatReplicate.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert response =~ "Colorful Threads"
    end

    @tag :live_call
    test "multi-message conversation" do
      # set_fake_llm_response({:ok, Message.new_assistant("\n\nRainbow Sox Co.")})

      # https://js.langchain.com/docs/modules/models/chat/
      {:ok, version} = ChatReplicate.latest_version(@live_call_model)
      {:ok, chat} = ChatReplicate.new(%{model: @live_call_model, version: version})

      {:ok, %Message{role: :assistant, content: response}} =
        ChatReplicate.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'."),
          Message.new_assistant!("Colorful Threads"),
          Message.new_user!("Ok, now return the same two words, but uppercase."),
        ])

      assert response =~ "COLORFUL THREADS"
    end

    @tag :live_call
    test "executes callback function when data is NOT streamed" do
      callback = fn %Message{} = new_message ->
        send(self(), {:message_received, new_message})
      end

      # https://js.langchain.com/docs/modules/models/chat/
      # NOTE streamed. Should receive complete message.
      {:ok, version} = ChatReplicate.latest_version(@live_call_model)
      {:ok, chat} = ChatReplicate.new(%{model: @live_call_model, version: version, stream: false})

      {:ok, message} =
        ChatReplicate.call(
          chat,
          [
            Message.new_user!("Return the response 'Hi'.")
          ],
          [],
          callback
        )

      assert message.content =~ "Hi"
      assert_receive {:message_received, received_item}, 500
      assert %Message{} = received_item
      assert received_item.role == :assistant
      assert received_item.content =~ "Hi"
    end

    @tag :live_call
    test "handles when request is too large" do
      {:ok, version} = ChatReplicate.latest_version(@live_call_model)

      {:ok, chat} =
        ChatReplicate.new(%{
          model: @live_call_model,
          version: version,
          stream: false,
          temperature: 1
        })

      {:error, reason} = ChatReplicate.call(chat, [too_large_user_request()])
      assert reason =~ "Your input is too long."
    end
  end

  describe "unsupported features for ChatReplicate" do
    test "streaming not supported" do
      {:error, changeset} =
        ChatReplicate.new(%{model: @live_call_model, version: "aabbccdd", stream: true})

      refute changeset.valid?
      assert {"streaming is currently unsupported for Replicate", _} = changeset.errors[:stream]
    end

    @tag :live_call
    test "function calls not supported" do
      {:ok, hello_world} =
        LangChain.Function.new(%{
          name: "hello_world",
          description: "Give a hello world greeting.",
          function: fn -> IO.puts("Hello world!") end
        })

      {:ok, version} = ChatReplicate.latest_version(@live_call_model)

      {:ok, chat} =
        ChatReplicate.new(%{model: @live_call_model, version: version})

      {:ok, message} =
        Message.new_user("Only using the functions you have been provided with, give a greeting.")

      assert_raise LangChain.LangChainError,
                   "Function calls are not currently supported for Replicate-hosted models",
                   fn ->
                     {:ok, _message} = ChatReplicate.call(chat, [message], [hello_world])
                   end
    end
  end

  describe "do_process_response/1" do
    test "handles receiving a message" do
      # this is a real response
      response = %{
        "id" => "bgntbjdbsrceapja7e7b6oxbym",
        "version" => "ac944f2e49c55c7e965fc3d93ad9a7d9d947866d6793fb849dd6b4747d0c061c",
        "input" => %{
          "prompt" => "Ben says Hi, what do you say?"
        },
        "logs" =>
          "Your formatted prompt is:\n[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\nBen says Hi, what do you say? [/INST]\nNot using LoRA\nExllama: True\nhostname: model-ac944f2e-4df82390750af308-gpu-a40-765fd8997c-xpbwb\n",
        "output" => [
          "",
          " Hello",
          " Ben",
          "!",
          " *",
          "sm",
          "iling",
          "*"
        ],
        "error" => nil,
        "status" => "succeeded",
        "created_at" => "2023-10-25T03:53:38.005549Z",
        "started_at" => "2023-10-25T03:53:38.104184Z",
        "completed_at" => "2023-10-25T03:53:38.679486Z",
        "metrics" => %{
          "predict_time" => 0.575302
        },
        "urls" => %{
          "cancel" =>
            "https ://api.replicate.com/v1/predictions/bgntbjdbsrceapja7e7b6oxbym/cancel",
          "get" => "https://api.replicate.com/v1/predictions/bgntbjdbsrceapja7e7b6oxbym"
        }
      }

      assert %Message{} = struct = ChatReplicate.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == " Hello Ben! *smiling*"
    end
  end
end
