Invoke-Expression "docker pull cxy201/noetic-vizor"

$remoteport = bash.exe -c "ifconfig eth0 | grep 'inet '"
$found = $remoteport -match '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}';

if ($found) {
  $remoteport = $matches[0];
  Write-Output $matches[0];
}
else {
  Write-Output "IP address could not be found";
  exit;
}

$ports = @(10000, 10001, 10002, 10003, 11311, 9090); #TCP, ros master, rosbridge websocket

for ($i = 0; $i -lt $ports.length; $i++) {
  $port = $ports[$i];
  Write-Output "opening $port...";

  Invoke-Expression "netsh interface portproxy delete v4tov4 listenport=$port";
  Invoke-Expression "netsh advfirewall firewall delete rule name=$port";
  # Write-Output "removed rules";

  Invoke-Expression "netsh interface portproxy add v4tov4 listenport=$port connectport=$port connectaddress=$remoteport";
  Invoke-Expression "netsh advfirewall firewall add rule name=$port dir=in action=allow protocol=TCP localport=$port";
}

Invoke-Expression "netsh interface portproxy show v4tov4";

Invoke-Expression "docker-compose -f vizor_config.yml up"
