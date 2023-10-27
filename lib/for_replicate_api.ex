defprotocol LangChain.ForReplicateApi do
  @moduledoc """
  A protocol that defines a way for converting the LangChain Elixir data structs
  to an [Replicate](https://replicate.com) supported data structure and format for making an API call.
  """

  @doc """
  Protocol callback function for converting different structs into a form that
  can be passed to the Replicate API.
  """
  @spec for_api(struct()) :: nil | %{String.t() => any()}
  def for_api(struct)
end

defimpl LangChain.ForReplicateApi, for: Any do
  def for_api(_struct), do: nil
end
