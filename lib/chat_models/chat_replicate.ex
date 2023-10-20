defmodule LangChain.ChatModels.ChatReplicate do
  @moduledoc """
  Represents a [Replicate](https://replicate.com)-hosted chat model.

  Parses and validates inputs for making a requests from the Replicate API.

  Converts responses into more specialized `LangChain` data structures.

  ## Supported models

  Replicate hosts many different types of AI models (not just LLMs) and not all
  models hosted on Replicate are available.

  Currently `LangChain.ChatModels.ChatReplicate` supports the following models:

  - [llama-2-7b-chat](https://replicate.com/meta/llama-2-7b-chat)
  - [llama-2-13b-chat](https://replicate.com/meta/llama-2-13b-chat)
  - [llama-2-70b-chat](https://replicate.com/meta/llama-2-70b-chat)

  Coming soon:

  - [mistral-7b-openorca](https://replicate.com/nateraw/mistral-7b-openorca)
  - [qwen-7b-chat](https://replicate.com/niron1/qwen-7b-chat)

  ## Caveats

  Replicate-hosted chains do not currently support all the features provided in
  `LangChain.ChatModels.ChatOpenAI`. In particular:

  - Some Replicate-hosted models [do support
    streaming](https://replicate.com/docs/streaming), but only to a separate,
    user-provided "callback" URL. This functionality is currently unsupported in
    Elixir LangChain. Any attempt to create a
    `%LangChain.ChatModels.ChatReplicate{}` struct with `stream: true` will
    return {:error, changeset}

  - Replicate-hosted LLMs do not currently support function calls. Any attempt
    to run a `LangChain.Chains.LLMChain` containing a function call will raise a
    `LangChain.LangChainError`.

  """

  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  import LangChain.Utils.ApiOverride
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.Message
  alias LangChain.LangChainError
  alias LangChain.ForReplicateApi
  alias LangChain.MessageDelta
  alias LangChain.Utils

  # NOTE: As of gpt-4 and gpt-3.5, only one function_call is issued at a time
  # even when multiple requests could be issued based on the prompt.

  # the actual response time for `wait_on_prediction_output/1` could be more
  # than this, because it's actually a recursive call to the status GET endpoint
  @receive_timeout 30_000

  @primary_key false
  embedded_schema do
    field(:endpoint, :string, default: "https://api.replicate.com/v1/")
    field(:model, :string, default: "meta/llama-2-7b-chat")

    # all Replicate-hosted models have a specific
    # [version](https://replicate.com/docs/how-does-replicate-work#versions)
    field(:version, :string)

    # Adjusts randomness of outputs, greater than 1 is random and 0 is
    # deterministic, 0.75 is a good starting value.
    field(:temperature, :float, default: 1.0)

    # When decoding text, samples from the top p percentage of most likely
    # tokens; lower to ignore less likely tokens.
    field(:top_p, :float, default: 0.9)

    # When decoding text, samples from the top k most likely tokens; lower to
    # ignore less likely token
    field(:top_k, :integer, default: 50)

    # Duration in seconds for the response to be received. When streaming a very
    # lengthy response, a longer time limit may be required. However, when it
    # goes on too long by itself, it tends to hallucinate more.
    field(:receive_timeout, :integer, default: @receive_timeout)

    # NOTE: stream: true is currently unsupported for Replicate
    field(:stream, :boolean, default: false)
  end

  @type t :: %ChatReplicate{}

  @type call_response :: {:ok, Message.t() | [Message.t()]} | {:error, String.t()}
  @type callback_data ::
          {:ok, Message.t() | MessageDelta.t() | [Message.t() | MessageDelta.t()]}
          | {:error, String.t()}

  @create_fields [:model, :version, :temperature, :top_p, :top_k, :stream, :receive_timeout]
  @required_fields [:model, :version]

  def put_auth_header(req) do
    # if no API key is set default to `""` which will raise a Stripe API error
    token = Config.resolve(:replicate_key, "")
    Req.Request.put_header(req, "authorization", "Token #{token}")
  end

  @doc """
  Setup a ChatReplicate client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatReplicate{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatReplicate client configuration and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, chain} ->
        chain

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_number(:temperature, greater_than_or_equal_to: 0, less_than_or_equal_to: 2)
    |> validate_number(:top_p, greater_than_or_equal_to: 0, less_than_or_equal_to: 1)
    |> validate_number(:top_k, greater_than_or_equal_to: 1, less_than_or_equal_to: 1000)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
    |> validate_change(:stream, fn field, value ->
      case value do
        false -> []
        _ -> [{field, "streaming is currently unsupported for Replicate"}]
      end
    end)
  end

  @doc """
  Return the params formatted for an API request.
  """
  @spec for_api(t, message :: [map()], functions :: [map()]) :: %{atom() => any()}
  def for_api(%ChatReplicate{} = replicate, messages, _functions) do
    {system_messages, chat_messages} =
      Enum.split_with(messages, fn msg -> msg.role == :system end)

    %{
      version: replicate.version,
      input: %{
        temperature: replicate.temperature,
        top_p: replicate.top_p,
        top_k: replicate.top_k,
        system_prompt: List.first(system_messages, ""),
        prompt: chat_messages |> Enum.map(&ForReplicateApi.for_api/1) |> Enum.join("\n")
      }
    }
  end

  @doc """
  Calls the Replicate API passing the ChatReplicate struct with configuration, plus
  either a simple message or the list of messages to act as the prompt.

  Optionally pass in a list of functions available to the LLM for requesting
  execution in response.

  Optionally pass in a callback function that can be executed as data is
  received from the API.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`. The `ChatReplicate` module is more focused on
  translating the `LangChain` data structures to and from the Replicate API.

  Another benefit of using `LangChain.Chains.LLMChain` is that it combines the
  storage of messages, adding functions, adding custom context that should be
  passed to functions, and automatically applying `LangChain.MessageDelta`
  structs as they are are received, then converting those to the full
  `LangChain.Message` once fully complete.
  """
  @spec call(
          t(),
          String.t() | [Message.t()],
          [LangChain.Function.t()],
          nil | (Message.t() | MessageDelta.t() -> any())
        ) :: call_response()
  def call(replicate, prompt, functions \\ [], callback_fn \\ nil)

  def call(%ChatReplicate{}, _messages, functions, _callback_fn) when functions != [] do
    raise LangChainError, "Function calls are not currently supported for Replicate-hosted models"
  end

  def call(%ChatReplicate{} = replicate, prompt, functions, callback_fn) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(replicate, messages, functions, callback_fn)
  end

  def call(%ChatReplicate{} = replicate, messages, functions, callback_fn)
      when is_list(messages) do
    if override_api_return?() do
      Logger.warning("Found override API response. Will not make live API call.")

      case get_api_override() do
        {:ok, {:ok, data} = response} ->
          # fire callback for fake responses too
          fire_callback(replicate, data, callback_fn)
          response

        _other ->
          raise LangChainError,
                "An unexpected fake API response was set. Should be an `{:ok, value}`"
      end
    else
      try do
        # make base api request and perform high-level success/failure checks
        case do_api_request(replicate, messages, functions, callback_fn) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err.message}
      end
    end
  end

  # Make the API request from the Replicate server.
  #
  # The result of the function is:
  #
  # - `result` - where `result` is a data-structure like a list or map.
  # - `{:error, reason}` - Where reason is a string explanation of what went wrong.

  # Executes the callback function passing the response only parsed to the data
  # structures.
  @doc false
  @spec do_api_request(t(), [Message.t()], [Function.t()], (any() -> any())) ::
          list() | struct() | {:error, String.t()}
  def do_api_request(%ChatReplicate{stream: false} = replicate, messages, functions, callback_fn) do
    with {:ok, prediction_id} <- create_prediction(replicate, messages, functions),
         {:ok, response} <- wait_on_prediction_output(prediction_id) do
      result = do_process_response(response)
      fire_callback(replicate, result, callback_fn)
      result
    end
  end

  # fire the callback if present.
  @spec fire_callback(
          t(),
          data :: callback_data() | [callback_data()],
          (callback_data() -> any())
        ) :: :ok
  defp fire_callback(%ChatReplicate{}, _data, nil), do: :ok

  defp fire_callback(%ChatReplicate{}, data, callback_fn) when is_function(callback_fn) do
    # OPTIONAL: Execute callback function
    callback_fn.(data)

    :ok
  end

  @doc false
  @spec do_process_response(data :: %{String.t() => any()}) :: Message.t() | {:error, String.t()}
  def do_process_response(%{"output" => output, "status" => "succeeded"}) do
    case Message.new(%{
           "role" => "assistant",
           "status" => "complete",
           "content" => Enum.join(output),
         }) do
      {:ok, message} ->
        message

      {:error, changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}
    end
  end

  # if "status" isn't "succeeded", then return the error
  def do_process_response(%{"error" => error}) do
    {:error, error}
  end

  # helper functions for working with the Replicate API

  @spec latest_version(model_id :: String.t()) :: {:ok, String.t()} | {:error, String.t()}
  def latest_version(model_id) do
    Req.new(base_url: "https://api.replicate.com/v1/")
    |> put_auth_header()
    |> Req.get(url: "models/#{model_id}/versions")
    |> case do
      {:ok, %Req.Response{body: %{"results" => results}}} -> {:ok, List.first(results)["id"]}
      {:error, reason} -> {:error, reason}
    end
  end

  defp create_prediction(replicate, messages, functions) do
    Req.new(base_url: "https://api.replicate.com/v1")
    |> put_auth_header()
    |> Req.post(
      url: "/predictions",
      receive_timeout: replicate.receive_timeout,
      json: for_api(replicate, messages, functions)
    )
    |> case do
      {:ok, %Req.Response{body: %{"id" => id}}} -> {:ok, id}
      {:error, error} -> {:error, error.message}
    end
  end

  # turn the async Replicate API into a normal (sync) Elixir function call -
  # this is what the official replicate-elixir client does as well:
  # https://github.com/replicate/replicate-elixir/blob/6c8e9660e0c579fa6025ca6046a09ea54c99ba4d/lib/client.ex#L41
  defp wait_on_prediction_output(prediction_id) do
    {:ok, %Req.Response{body: body}} =
      Req.new(base_url: "https://api.replicate.com/v1/")
      |> put_auth_header()
      |> Req.get(url: "predictions/#{prediction_id}")

    case body["status"] do
      "succeeded" -> {:ok, body}

      "failed" -> {:error, body["error"]}

      "canceled" -> {:error, "Prediction canceled"}

      status when status in ~w(starting processing) ->
        # recur and re-poll the API
        wait_on_prediction_output(prediction_id)
    end
  end
end
