#[int[]]� $ids;
#. .\Untitled1.ps1; InitializeIds 5; Connected -a 1 -b 4

Function InitializeIds([int] $size) 
{
��� $global:ids = New-Object int[] $size;
��� for($i = 0; $i -lt $size ; $i++)
��� {
������� $global:ids[$i] = $i
������� Write-Host $global:ids[$i];
��� }
}
Function Connected([int] $a, [int] $b)
{
��� return $global:ids[$a] -eq $global:ids[$b];
}

Function Union ([int] $a, [int] $b)
{
��� #save off the old value to make sure we update everyone that has this value. 
��� $idCurrentForA = $global:ids[$a];
���� 
�� 
��� for($i = 0; $i -lt $global:Ids.Length ; $i++)
��� {
������� if($global:ids[$i] -eq $idCurrentForA )
������� {
���������� $global:ids[$i] = $b;����� 
������� }
��� }
}
Function print( ) {
� 
��� for($i = 0; $i -lt $global:Ids.Length ; $i++)
��� { 
������� write-host $global:ids[$i];
��� }
}

Function print24( ) {
� 
��� for($i = 0; $i -lt 24)
��� { 
������ 
������� for($ix = 1; $ix -le 4)
������� {
����������� 
����������� $ele=� $i+ $ix;
����������� write-host $ele;
����������� $ix=$ix+1;
������� }
�������� $i=($i+4);
�������� write-host "new line";
��� }
}
