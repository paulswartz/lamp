# Script for populating the database. You can run it as:
#
#     mix run priv/repo/seeds.exs
#
# For now, this runs the Python seed script, but configured to use the current
# database.
db_config = Application.get_env(:api, Api.Repo)

{output, result} =
  System.shell(
    "poetry run seed_metadata --clear-static --seed-file tests/test_files/july_17_filepaths.json",
    cd: "../python_src",
    stderr_to_stdout: true,
    env: %{
      "BOOTSTRAPPED" => "1",
      "DB_HOST" => db_config[:hostname],
      "DB_PORT" => db_config[:port],
      "DB_NAME" => db_config[:database],
      "DB_USER" => db_config[:username],
      "DB_PASSWORD" => db_config[:password]
    }
  )

if result != 0 do
  IO.puts(output)
  exit(result)
end
